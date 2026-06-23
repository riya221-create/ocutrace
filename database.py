"""
OcuTrace — PostgreSQL Database Module
=======================================
Stores clinical reports and RAG evidence results per patient.

Schema:
    patients          — patient profiles
    scan_sessions     — per-visit OCT session metadata
    clinical_reports  — LLM-generated clinical reports per session
    rag_evidence      — PubMed RAG results per session
    biomarker_records — quantitative biomarker values per session

Setup:
    1. Install PostgreSQL and create a database:
       createdb ocutrace

    2. Set environment variable:
       export DATABASE_URL=postgresql://user:password@localhost:5432/ocutrace

    3. Run migrations:
       python database.py --migrate

    4. Use in code:
       from database import OcuTraceDB
       db = OcuTraceDB()
       patient_id = db.create_patient("Riya Kashyap", dob="1990-05-12", rvo_type="BRVO")
       session_id = db.create_session(patient_id, visit_date="2024-01-14")
       db.save_clinical_report(session_id, report.to_dict())
       db.save_rag_evidence(session_id, rag_result.to_dict())
"""

import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.extensions import connection as PgConnection
except ImportError:
    raise ImportError(
        "psycopg2 not installed. Run: pip install psycopg2-binary"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_connection_string() -> str:
    """
    Get PostgreSQL connection string from environment.
    Supports both DATABASE_URL and individual env vars.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    host     = os.environ.get("DB_HOST",     "localhost")
    port     = os.environ.get("DB_PORT",     "5432")
    name     = os.environ.get("DB_NAME",     "ocutrace")
    user     = os.environ.get("DB_USER",     "postgres")
    password = os.environ.get("DB_PASSWORD", "")

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Patients ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name       TEXT NOT NULL,
    date_of_birth   DATE,
    rvo_type        TEXT CHECK (rvo_type IN ('BRVO', 'CRVO', 'Hemi-RVO', 'Unknown')),
    affected_eye    TEXT CHECK (affected_eye IN ('Left', 'Right', 'Both')),
    conditions      JSONB DEFAULT '[]',      -- comorbidities list
    medications     TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Scan sessions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scan_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_date      DATE NOT NULL,
    visit_number    INTEGER,                 -- auto-incremented per patient
    injections_since_last INTEGER DEFAULT 0,
    scan_type       TEXT DEFAULT 'OCT',
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Biomarker records ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS biomarker_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES scan_sessions(id) ON DELETE CASCADE,
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_date      DATE NOT NULL,

    -- Quantitative biomarkers
    crt_um          NUMERIC(8,2),     -- central retinal thickness µm
    irf_mm3         NUMERIC(8,4),     -- intraretinal fluid mm³
    srf_mm3         NUMERIC(8,4),     -- subretinal fluid mm³
    ped_mm3         NUMERIC(8,4),     -- pigment epithelial detachment mm³
    irf_pct         NUMERIC(6,2),     -- IRF as % of scan area
    srf_pct         NUMERIC(6,2),     -- SRF as % of scan area
    dril_pct        NUMERIC(6,2),     -- DRIL extent %
    ez_integrity    NUMERIC(5,4),     -- ellipsoid zone integrity 0-1

    -- Confidence scores
    overall_confidence   NUMERIC(4,3),
    confidence_level     TEXT CHECK (confidence_level IN ('high', 'moderate', 'low')),
    registration_quality NUMERIC(4,3),

    raw_biomarkers  JSONB,            -- full biomarker JSON for reference
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Clinical reports ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clinical_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES scan_sessions(id) ON DELETE CASCADE,
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_date      DATE NOT NULL,

    -- Report content
    summary         TEXT NOT NULL,
    risk_level      TEXT NOT NULL CHECK (risk_level IN ('low', 'moderate', 'high')),
    risk_rationale  TEXT,
    recommendation  TEXT,
    watch_next_visit TEXT,

    -- Deltas vs previous visit
    delta_crt_um    NUMERIC(8,2),
    delta_irf_pct   NUMERIC(6,2),
    delta_dril_pct  NUMERIC(6,2),

    -- Generation metadata
    model_used      TEXT DEFAULT 'rule-based',
    raw_report      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── RAG evidence ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_evidence (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES scan_sessions(id) ON DELETE CASCADE,
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    visit_date      DATE NOT NULL,

    -- Patient inputs
    symptoms        TEXT,
    condition       TEXT,

    -- RAG outputs
    summary         TEXT,
    risk_flags      JSONB DEFAULT '[]',
    doctor_questions JSONB DEFAULT '[]',
    references      JSONB DEFAULT '[]',   -- list of {pmid, citation, url}
    pubmed_query    TEXT,
    articles_count  INTEGER DEFAULT 0,

    -- Metadata
    model_used      TEXT DEFAULT 'claude-sonnet-4-6',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sessions_patient
    ON scan_sessions(patient_id, visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_biomarkers_patient
    ON biomarker_records(patient_id, visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_reports_patient
    ON clinical_reports(patient_id, visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_rag_patient
    ON rag_evidence(patient_id, visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_reports_risk
    ON clinical_reports(risk_level, created_at DESC);
"""


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OcuTraceDB:
    """
    PostgreSQL database interface for OcuTrace.

    Handles patient profiles, scan sessions, biomarker records,
    clinical reports, and RAG evidence results.

    Usage:
        db = OcuTraceDB()
        patient_id = db.create_patient("Jane Doe", rvo_type="BRVO")
        session_id = db.create_session(patient_id, visit_date="2024-01-14")
        db.save_clinical_report(session_id, report.to_dict())
        history = db.get_patient_history(patient_id)
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.conn_str = connection_string or get_connection_string()
        self._conn: Optional[PgConnection] = None
        self._connect()

    def _connect(self):
        try:
            self._conn = psycopg2.connect(
                self.conn_str,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
            self._conn.autocommit = False
            print(f"[OcuTrace DB] Connected to PostgreSQL")
        except psycopg2.OperationalError as e:
            raise ConnectionError(
                f"Cannot connect to PostgreSQL.\n"
                f"Check DATABASE_URL or DB_HOST/DB_NAME/DB_USER/DB_PASSWORD env vars.\n"
                f"Error: {e}"
            )

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor with auto-commit/rollback."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def migrate(self):
        """Run schema migrations — safe to call multiple times (IF NOT EXISTS)."""
        with self._cursor() as cur:
            cur.execute(SCHEMA_SQL)
        print("[OcuTrace DB] Schema migrations complete.")

    def close(self):
        if self._conn:
            self._conn.close()

    # ── Patients ──────────────────────────────────────────────────────────────

    def create_patient(
        self,
        full_name:     str,
        date_of_birth: Optional[str] = None,   # "YYYY-MM-DD"
        rvo_type:      str = "Unknown",
        affected_eye:  str = "Left",
        conditions:    Optional[list] = None,
        medications:   Optional[str] = None,
        notes:         Optional[str] = None,
    ) -> str:
        """Create a new patient record. Returns patient UUID."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO patients
                    (full_name, date_of_birth, rvo_type, affected_eye,
                     conditions, medications, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id::text
            """, (
                full_name,
                date_of_birth,
                rvo_type,
                affected_eye,
                json.dumps(conditions or []),
                medications,
                notes,
            ))
            patient_id = cur.fetchone()["id"]
        print(f"[OcuTrace DB] Patient created: {full_name} ({patient_id})")
        return patient_id

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Fetch a patient by ID."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE id = %s", (patient_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def search_patients(self, name_query: str) -> list[dict]:
        """Search patients by name (case-insensitive partial match)."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT id::text, full_name, date_of_birth, rvo_type, affected_eye
                FROM patients
                WHERE full_name ILIKE %s
                ORDER BY full_name
                LIMIT 20
            """, (f"%{name_query}%",))
            return [dict(r) for r in cur.fetchall()]

    def list_patients(self, limit: int = 50) -> list[dict]:
        """List all patients ordered by most recently updated."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT p.id::text, p.full_name, p.rvo_type, p.affected_eye,
                       COUNT(s.id) AS session_count,
                       MAX(s.visit_date) AS last_visit
                FROM patients p
                LEFT JOIN scan_sessions s ON s.patient_id = p.id
                GROUP BY p.id, p.full_name, p.rvo_type, p.affected_eye
                ORDER BY last_visit DESC NULLS LAST
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ── Scan sessions ─────────────────────────────────────────────────────────

    def create_session(
        self,
        patient_id:             str,
        visit_date:             str,    # "YYYY-MM-DD"
        injections_since_last:  int = 0,
        scan_type:              str = "OCT",
        notes:                  Optional[str] = None,
    ) -> str:
        """Create a new scan session. Returns session UUID."""
        with self._cursor() as cur:
            # Auto-increment visit number for this patient
            cur.execute("""
                SELECT COALESCE(MAX(visit_number), 0) + 1
                FROM scan_sessions WHERE patient_id = %s
            """, (patient_id,))
            visit_number = cur.fetchone()["coalesce"]

            cur.execute("""
                INSERT INTO scan_sessions
                    (patient_id, visit_date, visit_number,
                     injections_since_last, scan_type, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id::text
            """, (patient_id, visit_date, visit_number,
                  injections_since_last, scan_type, notes))
            session_id = cur.fetchone()["id"]
        print(f"[OcuTrace DB] Session created: visit {visit_number} on {visit_date}")
        return session_id

    def get_sessions(self, patient_id: str) -> list[dict]:
        """Get all sessions for a patient, newest first."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT id::text, visit_date, visit_number,
                       injections_since_last, scan_type, notes, created_at
                FROM scan_sessions
                WHERE patient_id = %s
                ORDER BY visit_date DESC
            """, (patient_id,))
            return [dict(r) for r in cur.fetchall()]

    # ── Biomarker records ─────────────────────────────────────────────────────

    def save_biomarkers(
        self,
        session_id:    str,
        patient_id:    str,
        visit_date:    str,
        biomarkers:    dict,
        confidence:    Optional[dict] = None,
    ) -> str:
        """Save biomarker record for a session. Returns record UUID."""
        conf = confidence or {}
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO biomarker_records (
                    session_id, patient_id, visit_date,
                    crt_um, irf_mm3, srf_mm3, ped_mm3,
                    irf_pct, srf_pct, dril_pct, ez_integrity,
                    overall_confidence, confidence_level, registration_quality,
                    raw_biomarkers
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING id::text
            """, (
                session_id, patient_id, visit_date,
                biomarkers.get("crt_um"),
                biomarkers.get("irf_mm3"),
                biomarkers.get("srf_mm3"),
                biomarkers.get("ped_mm3"),
                biomarkers.get("irf_pct"),
                biomarkers.get("srf_pct"),
                biomarkers.get("dril_pct"),
                biomarkers.get("ez_integrity"),
                conf.get("overall_confidence"),
                conf.get("overall_level"),
                conf.get("registration_quality"),
                json.dumps(biomarkers),
            ))
            return cur.fetchone()["id"]

    def get_biomarker_history(self, patient_id: str) -> list[dict]:
        """Get full biomarker history for a patient, oldest first."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT visit_date, crt_um, irf_mm3, srf_mm3,
                       dril_pct, ez_integrity, overall_confidence, confidence_level
                FROM biomarker_records
                WHERE patient_id = %s
                ORDER BY visit_date ASC
            """, (patient_id,))
            return [dict(r) for r in cur.fetchall()]

    # ── Clinical reports ──────────────────────────────────────────────────────

    def save_clinical_report(
        self,
        session_id:  str,
        patient_id:  str,
        visit_date:  str,
        report:      dict,
        deltas:      Optional[dict] = None,
        model_used:  str = "claude-sonnet-4-6",
    ) -> str:
        """
        Save a clinical report for a session.

        Args:
            session_id:  UUID of the scan session
            patient_id:  UUID of the patient
            visit_date:  "YYYY-MM-DD"
            report:      ClinicalReport.to_dict() output
            deltas:      biomarker_deltas dict from diff engine
            model_used:  model name or "rule-based"

        Returns:
            UUID of the saved report
        """
        d = deltas or {}
        delta_crt  = d.get("crt_um",   {}).get("delta_abs")
        delta_irf  = d.get("irf_pct",  {}).get("delta_abs")
        delta_dril = d.get("dril_pct", {}).get("delta_abs")

        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO clinical_reports (
                    session_id, patient_id, visit_date,
                    summary, risk_level, risk_rationale,
                    recommendation, watch_next_visit,
                    delta_crt_um, delta_irf_pct, delta_dril_pct,
                    model_used, raw_report
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING id::text
            """, (
                session_id, patient_id, visit_date,
                report.get("summary", ""),
                report.get("risk_level", "moderate"),
                report.get("risk_rationale", ""),
                report.get("recommendation", ""),
                report.get("watch_next_visit", ""),
                delta_crt, delta_irf, delta_dril,
                model_used,
                json.dumps(report),
            ))
            report_id = cur.fetchone()["id"]
        print(f"[OcuTrace DB] Clinical report saved: {report_id}")
        return report_id

    def get_clinical_reports(self, patient_id: str) -> list[dict]:
        """Get all clinical reports for a patient, newest first."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT id::text, visit_date, summary, risk_level,
                       recommendation, watch_next_visit,
                       delta_crt_um, delta_irf_pct, delta_dril_pct,
                       model_used, created_at
                FROM clinical_reports
                WHERE patient_id = %s
                ORDER BY visit_date DESC
            """, (patient_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_latest_report(self, patient_id: str) -> Optional[dict]:
        """Get the most recent clinical report for a patient."""
        reports = self.get_clinical_reports(patient_id)
        return reports[0] if reports else None

    # ── RAG evidence ──────────────────────────────────────────────────────────

    def save_rag_evidence(
        self,
        session_id:  str,
        patient_id:  str,
        visit_date:  str,
        rag_result:  dict,
        symptoms:    Optional[str] = None,
        condition:   Optional[str] = None,
        pubmed_query: Optional[str] = None,
    ) -> str:
        """
        Save RAG evidence result for a session.

        Args:
            rag_result:  RAGResult.to_dict() output
            symptoms:    Raw symptom string submitted
            condition:   Disease condition queried

        Returns:
            UUID of the saved evidence record
        """
        refs = rag_result.get("references", [])
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO rag_evidence (
                    session_id, patient_id, visit_date,
                    symptoms, condition, summary,
                    risk_flags, doctor_questions, references,
                    pubmed_query, articles_count
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING id::text
            """, (
                session_id, patient_id, visit_date,
                symptoms, condition,
                rag_result.get("summary", ""),
                json.dumps(rag_result.get("risk_flags", [])),
                json.dumps(rag_result.get("doctor_questions", [])),
                json.dumps(refs),
                pubmed_query,
                len(refs),
            ))
            rag_id = cur.fetchone()["id"]
        print(f"[OcuTrace DB] RAG evidence saved: {rag_id} ({len(refs)} references)")
        return rag_id

    def get_rag_evidence(self, patient_id: str) -> list[dict]:
        """Get all RAG evidence records for a patient, newest first."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT id::text, visit_date, symptoms, condition,
                       summary, risk_flags, doctor_questions,
                       references, articles_count, created_at
                FROM rag_evidence
                WHERE patient_id = %s
                ORDER BY visit_date DESC
            """, (patient_id,))
            return [dict(r) for r in cur.fetchall()]

    # ── Patient history (full) ────────────────────────────────────────────────

    def get_patient_history(self, patient_id: str) -> dict:
        """
        Get complete patient history: profile + sessions + biomarkers +
        clinical reports + RAG evidence.

        Returns a single dict with all records grouped.
        """
        patient  = self.get_patient(patient_id)
        if not patient:
            raise ValueError(f"Patient not found: {patient_id}")

        return {
            "patient":          patient,
            "sessions":         self.get_sessions(patient_id),
            "biomarkers":       self.get_biomarker_history(patient_id),
            "clinical_reports": self.get_clinical_reports(patient_id),
            "rag_evidence":     self.get_rag_evidence(patient_id),
        }

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_risk_summary(self) -> dict:
        """Aggregate risk level counts across all patients (latest report per patient)."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT risk_level, COUNT(*) as count
                FROM (
                    SELECT DISTINCT ON (patient_id)
                        patient_id, risk_level
                    FROM clinical_reports
                    ORDER BY patient_id, visit_date DESC
                ) latest
                GROUP BY risk_level
            """)
            rows = cur.fetchall()
            return {r["risk_level"]: r["count"] for r in rows}

    def get_high_risk_patients(self) -> list[dict]:
        """Get all patients whose latest report is high risk."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT p.id::text, p.full_name, p.rvo_type,
                       r.visit_date, r.summary, r.recommendation
                FROM patients p
                JOIN (
                    SELECT DISTINCT ON (patient_id)
                        patient_id, visit_date, summary, recommendation, risk_level
                    FROM clinical_reports
                    ORDER BY patient_id, visit_date DESC
                ) r ON r.patient_id = p.id
                WHERE r.risk_level = 'high'
                ORDER BY r.visit_date DESC
            """)
            return [dict(row) for row in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT HELPER — patient selector sidebar component
# ─────────────────────────────────────────────────────────────────────────────

def render_patient_selector(db: OcuTraceDB) -> Optional[str]:
    """
    Render a patient search + select widget for Streamlit.
    Returns selected patient_id or None.

    Usage in app.py:
        from database import OcuTraceDB, render_patient_selector
        db = OcuTraceDB()
        patient_id = render_patient_selector(db)
    """
    try:
        import streamlit as st
    except ImportError:
        return None

    st.markdown("#### Patient")
    search = st.text_input("Search patient", placeholder="Type name...")
    patients = db.search_patients(search) if search else db.list_patients(20)

    if not patients:
        st.caption("No patients found.")
        return None

    options = {f"{p['full_name']} ({p['rvo_type']})": p["id"] for p in patients}
    selected_label = st.selectbox("Select patient", list(options.keys()))
    return options.get(selected_label)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — migrations and demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OcuTrace database CLI")
    parser.add_argument("--migrate", action="store_true", help="Run schema migrations")
    parser.add_argument("--demo",    action="store_true", help="Insert demo patient data")
    parser.add_argument("--status",  action="store_true", help="Show database status")
    args = parser.parse_args()

    db = OcuTraceDB()

    if args.migrate:
        db.migrate()

    if args.demo:
        print("\n[OcuTrace DB] Inserting demo patient...")

        patient_id = db.create_patient(
            full_name     = "Demo Patient",
            date_of_birth = "1962-03-15",
            rvo_type      = "BRVO",
            affected_eye  = "Left",
            conditions    = ["Hypertension", "High cholesterol"],
            medications   = "amlodipine 5mg, atorvastatin 20mg",
        )

        session_id = db.create_session(
            patient_id            = patient_id,
            visit_date            = "2024-01-14",
            injections_since_last = 0,
        )

        db.save_biomarkers(
            session_id = session_id,
            patient_id = patient_id,
            visit_date = "2024-01-14",
            biomarkers = {
                "crt_um": 412.0, "irf_mm3": 2.3, "srf_mm3": 0.8,
                "ped_mm3": 0.1, "irf_pct": 4.5, "srf_pct": 1.6,
                "dril_pct": 18.2, "ez_integrity": 0.72,
            },
            confidence = {"overall_confidence": 0.82, "overall_level": "high",
                          "registration_quality": 0.95},
        )

        session_id_2 = db.create_session(
            patient_id            = patient_id,
            visit_date            = "2024-03-02",
            injections_since_last = 2,
        )

        report_id = db.save_clinical_report(
            session_id = session_id_2,
            patient_id = patient_id,
            visit_date = "2024-03-02",
            report = {
                "summary":          "CRT decreased 94µm following 2 anti-VEGF injections. DRIL increased 21%.",
                "risk_level":       "moderate",
                "risk_rationale":   "DRIL worsening despite edema improvement indicates ischemic progression.",
                "recommendation":   "Maintain 6-8 week injection interval.",
                "watch_next_visit": "EZ integrity and capillary non-perfusion on OCTA.",
            },
            deltas = {
                "crt_um":   {"delta_abs": -94.0},
                "irf_pct":  {"delta_abs": -2.7},
                "dril_pct": {"delta_abs": 3.9},
            },
        )

        rag_id = db.save_rag_evidence(
            session_id = session_id_2,
            patient_id = patient_id,
            visit_date = "2024-03-02",
            rag_result = {
                "summary":          "Literature supports anti-VEGF for BRVO-ME [PMID:34521234].",
                "risk_flags":       ["DRIL progression — associated with poor visual outcomes [PMID:33412198]"],
                "doctor_questions": ["Has visual acuity improved subjectively?",
                                     "Consider OCTA to assess capillary non-perfusion."],
                "references":       [{"pmid": "34521234", "citation": "Chen et al. (2021)", "url": "https://pubmed.ncbi.nlm.nih.gov/34521234/"}],
            },
            symptoms  = "blurred vision left eye, dark spot in central vision",
            condition = "BRVO",
        )

        print(f"\n✓ Demo patient created:  {patient_id}")
        print(f"✓ Session 1:             {session_id}")
        print(f"✓ Session 2:             {session_id_2}")
        print(f"✓ Clinical report:       {report_id}")
        print(f"✓ RAG evidence:          {rag_id}")

        history = db.get_patient_history(patient_id)
        print(f"\nPatient history: {len(history['sessions'])} sessions, "
              f"{len(history['clinical_reports'])} reports, "
              f"{len(history['rag_evidence'])} RAG records")

    if args.status:
        patients = db.list_patients(5)
        risk     = db.get_risk_summary()
        print(f"\n[OcuTrace DB] Status")
        print(f"  Patients:      {len(patients)}")
        print(f"  Risk summary:  {risk}")
        if patients:
            print(f"  Recent patients:")
            for p in patients:
                print(f"    {p['full_name']} ({p['rvo_type']}) — {p['session_count']} visits")

    db.close()
