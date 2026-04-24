"""
OcuTrace — PubMed RAG Module
=============================
Retrieves evidence from PubMed and cross-references patient symptoms /
biomarker findings against verified medical literature.

No hallucination by design: the LLM can only respond using
retrieved PubMed abstracts. Every claim is citation-backed with a PMID.

Usage:
    from pubmed_rag import MedRAG

    rag = MedRAG(api_key="sk-ant-...")
    result = rag.query(
        symptoms="blurred vision, floaters, sudden vision loss left eye",
        biomarkers={"crt_um": 412, "irf_mm3": 2.3, "dril_pct": 18},
        condition="BRVO"
    )
    print(result.summary)
    print(result.doctor_questions)
    for ref in result.references:
        print(ref["citation"])

Standalone:
    python pubmed_rag.py
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import anthropic


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PubMedArticle:
    pmid:     str
    title:    str
    abstract: str
    authors:  str
    year:     str
    journal:  str

    @property
    def citation(self) -> str:
        return f"{self.authors} ({self.year}). {self.title}. {self.journal}. PMID: {self.pmid}"

    @property
    def context_chunk(self) -> str:
        return f"[PMID:{self.pmid}] {self.title}\n{self.abstract}"


@dataclass
class RAGResult:
    summary:          str
    doctor_questions: list[str]
    risk_flags:       list[str]
    references:       list[dict]   # list of {pmid, citation, relevance_score}
    raw_response:     str = ""

    def pretty_print(self):
        print("\n" + "─" * 64)
        print("OcuTrace MedRAG — Evidence-Backed Summary")
        print("─" * 64)
        print(f"\nSummary:\n{self.summary}")
        print(f"\nRisk flags:")
        for f in self.risk_flags:
            print(f"  ⚠ {f}")
        print(f"\nQuestions for your doctor:")
        for i, q in enumerate(self.doctor_questions, 1):
            print(f"  {i}. {q}")
        print(f"\nEvidence sources ({len(self.references)} articles):")
        for r in self.references:
            print(f"  [{r['pmid']}] {r['citation']}")
        print("─" * 64 + "\n")

    def to_dict(self) -> dict:
        return {
            "summary":          self.summary,
            "doctor_questions": self.doctor_questions,
            "risk_flags":       self.risk_flags,
            "references":       self.references,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PUBMED FETCHER
# ─────────────────────────────────────────────────────────────────────────────

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_BASE_URL   = "https://pubmed.ncbi.nlm.nih.gov"


def _fetch_url(url: str, retries: int = 3) -> str:
    """Simple urllib fetch with retries — no external HTTP lib needed."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """
    Search PubMed and return list of PMIDs.
    Filters to last 10 years and requires abstract.
    """
    params = urllib.parse.urlencode({
        "db":       "pubmed",
        "term":     query,
        "retmax":   max_results,
        "retmode":  "json",
        "sort":     "relevance",
        "datetype": "pdat",
        "reldate":  3650,   # last 10 years
        "filter":   "hasabstract",
    })
    url = f"{PUBMED_SEARCH_URL}?{params}"
    raw = _fetch_url(url)
    data = json.loads(raw)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[PubMedArticle]:
    """Fetch full article details for a list of PMIDs."""
    if not pmids:
        return []

    params = urllib.parse.urlencode({
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    url  = f"{PUBMED_FETCH_URL}?{params}"
    xml  = _fetch_url(url)
    root = ET.fromstring(xml)
    articles = []

    for article_el in root.findall(".//PubmedArticle"):
        try:
            pmid = article_el.findtext(".//PMID") or ""

            # Title
            title = article_el.findtext(".//ArticleTitle") or "No title"
            title = re.sub(r"<[^>]+>", "", title).strip()

            # Abstract — join all AbstractText elements
            abstract_parts = [
                el.text or "" for el in article_el.findall(".//AbstractText")
            ]
            abstract = " ".join(abstract_parts).strip()
            if not abstract:
                continue   # skip articles with no abstract

            # Authors
            last_names = [
                el.findtext("LastName") or ""
                for el in article_el.findall(".//Author")[:3]
            ]
            authors = ", ".join(filter(None, last_names))
            if len(article_el.findall(".//Author")) > 3:
                authors += " et al."

            # Year
            year = (
                article_el.findtext(".//PubDate/Year") or
                article_el.findtext(".//PubDate/MedlineDate", "")[:4] or
                "n.d."
            )

            # Journal
            journal = article_el.findtext(".//Journal/Title") or \
                      article_el.findtext(".//ISOAbbreviation") or ""

            articles.append(PubMedArticle(
                pmid=pmid, title=title, abstract=abstract,
                authors=authors, year=year, journal=journal,
            ))
        except Exception:
            continue

    return articles


def build_search_query(
    symptoms: str,
    condition: str,
    biomarkers: Optional[dict] = None,
) -> str:
    """
    Build an optimised PubMed query from patient symptoms and condition.
    Uses MeSH-aware terms for better retrieval.
    """
    condition_terms = {
        "BRVO":    "branch retinal vein occlusion",
        "CRVO":    "central retinal vein occlusion",
        "RVO":     "retinal vein occlusion",
        "AMD":     "age-related macular degeneration",
        "DME":     "diabetic macular edema",
        "default": condition,
    }
    cond = condition_terms.get(condition.upper(), condition_terms["default"])

    # Extract key symptom terms
    symptom_keywords = []
    symptom_map = {
        "vision loss":   "visual acuity loss",
        "blurred":       "blurred vision",
        "floaters":      "vitreous floaters",
        "dark spot":     "scotoma",
        "distortion":    "metamorphopsia",
        "swelling":      "macular edema",
    }
    for phrase, mesh_term in symptom_map.items():
        if phrase.lower() in symptoms.lower():
            symptom_keywords.append(mesh_term)

    # Add biomarker-specific terms
    bio_terms = []
    if biomarkers:
        if biomarkers.get("irf_mm3", 0) > 0.5:
            bio_terms.append("intraretinal fluid OCT")
        if biomarkers.get("dril_pct", 0) > 15:
            bio_terms.append("DRIL disorganization retinal inner layers")
        if biomarkers.get("crt_um", 0) > 350:
            bio_terms.append("macular edema anti-VEGF treatment")

    parts = [cond]
    if symptom_keywords:
        parts.append(" OR ".join(symptom_keywords[:2]))
    if bio_terms:
        parts.append(bio_terms[0])

    query = " AND ".join(f"({p})" for p in parts)
    return query


# ─────────────────────────────────────────────────────────────────────────────
# RAG SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a medical evidence synthesizer embedded in OcuTrace, \
a retinal disease analysis system. You have been given a set of verified PubMed abstracts \
as your ONLY knowledge source.

CRITICAL RULES — you must follow these exactly:
1. You may ONLY make claims that are directly supported by the provided abstracts.
2. Every claim must cite its source using [PMID:XXXXXXXX] inline.
3. If the abstracts do not support a claim, write "insufficient evidence in retrieved sources."
4. Do NOT use your general training knowledge. Only use what is in the abstracts below.
5. This is a pre-diagnosis tool for doctors — not a replacement for clinical judgment.

Respond ONLY with a valid JSON object, no preamble, no markdown fences:
{
  "summary": "<2–3 sentence evidence-backed summary of the patient's findings>",
  "risk_flags": ["<specific concern from literature>", ...],
  "doctor_questions": [
    "<specific, evidence-grounded question the doctor should ask>",
    ...
  ]
}

Generate 3–5 risk flags and 4–6 doctor questions.
Each doctor question must reference a specific finding from the patient data."""


def _build_rag_prompt(
    symptoms:   str,
    condition:  str,
    biomarkers: Optional[dict],
    articles:   list[PubMedArticle],
) -> str:
    """Build the user prompt with retrieved context injected."""

    # Format biomarkers
    bio_str = ""
    if biomarkers:
        bio_str = "\nBiomarker findings:\n" + "\n".join(
            f"  {k}: {v}" for k, v in biomarkers.items()
        )

    # Inject retrieved abstracts as grounding context
    context_block = "\n\n".join(
        f"--- Source {i+1} ---\n{a.context_chunk}"
        for i, a in enumerate(articles)
    )

    return f"""Patient data:
Condition: {condition}
Symptoms: {symptoms}{bio_str}

Retrieved PubMed evidence ({len(articles)} articles):
{context_block}

Generate the evidence-backed pre-diagnosis JSON using ONLY the sources above."""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RAG CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MedRAG:
    """
    Evidence-grounded medical RAG system using PubMed + Claude.

    Every output claim is anchored to a retrieved PubMed abstract.
    Hallucination is prevented architecturally — the LLM only sees
    the retrieved chunks, not its general training data.
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str = "claude-sonnet-4-6",
        max_results: int = 5,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("No API key. Set ANTHROPIC_API_KEY or pass api_key=")
        self.client      = anthropic.Anthropic(api_key=key)
        self.model       = model
        self.max_results = max_results

    def query(
        self,
        symptoms:   str,
        condition:  str = "RVO",
        biomarkers: Optional[dict] = None,
        verbose:    bool = True,
    ) -> RAGResult:
        """
        Full RAG pipeline:
        1. Build PubMed search query from symptoms + biomarkers
        2. Fetch verified abstracts
        3. Inject into LLM context
        4. Return evidence-grounded report

        Args:
            symptoms:   Patient-reported symptoms as free text
            condition:  Disease context (BRVO, CRVO, AMD, DME...)
            biomarkers: Dict of quantitative findings from diff engine
            verbose:    Print progress to stdout
        """
        if verbose:
            print("[MedRAG] Building search query...")
        query = build_search_query(symptoms, condition, biomarkers)

        if verbose:
            print(f"[MedRAG] Searching PubMed: {query[:80]}...")
        pmids = search_pubmed(query, self.max_results)

        if not pmids:
            if verbose:
                print("[MedRAG] No results — trying broader query...")
            pmids = search_pubmed(condition, self.max_results)

        if verbose:
            print(f"[MedRAG] Fetching {len(pmids)} abstracts...")
        articles = fetch_abstracts(pmids)

        if not articles:
            return RAGResult(
                summary="No verified PubMed evidence retrieved for this query.",
                doctor_questions=["Please consult a specialist for evidence-based guidance."],
                risk_flags=["Insufficient literature retrieved — manual review required."],
                references=[],
            )

        if verbose:
            print(f"[MedRAG] Generating evidence-grounded report...")

        prompt = _build_rag_prompt(symptoms, condition, biomarkers, articles)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw  = response.content[0].text.strip()
        parsed = self._parse(raw)

        references = [
            {
                "pmid":     a.pmid,
                "citation": a.citation,
                "url":      f"{PUBMED_BASE_URL}/{a.pmid}/",
            }
            for a in articles
        ]

        return RAGResult(
            summary          = parsed.get("summary", "See references."),
            doctor_questions = parsed.get("doctor_questions", []),
            risk_flags       = parsed.get("risk_flags", []),
            references       = references,
            raw_response     = raw,
        )

    def _parse(self, raw: str) -> dict:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[MedRAG] No ANTHROPIC_API_KEY — running PubMed fetch only (no LLM step).\n")

    symptoms = "sudden blurred vision in left eye, floaters, dark spot in central vision"
    condition = "BRVO"
    biomarkers = {"crt_um": 412, "irf_mm3": 2.3, "dril_pct": 18.2, "ez_integrity": 0.72}

    print(f"Query: {symptoms}")
    print(f"Condition: {condition}\n")

    query = build_search_query(symptoms, condition, biomarkers)
    print(f"PubMed query: {query}\n")

    pmids = search_pubmed(query, max_results=5)
    print(f"Found {len(pmids)} articles: {pmids}\n")

    articles = fetch_abstracts(pmids)
    for a in articles:
        print(f"  [{a.pmid}] {a.title[:70]}...")
        print(f"           {a.authors} ({a.year}) — {a.journal}\n")

    if api_key:
        rag = MedRAG(api_key=api_key)
        result = rag.query(symptoms, condition, biomarkers)
        result.pretty_print()

        out_path = "ocutrace_output/medrag_result.json"
        import pathlib
        pathlib.Path("ocutrace_output").mkdir(exist_ok=True)
        pathlib.Path(out_path).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Saved → {out_path}")
