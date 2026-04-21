# OcuTrace

OcuTrace is a Streamlit app for longitudinal OCT progression analysis in retinal vein occlusion. It compares two scans, generates biomarker deltas, renders overlays, and can optionally produce an Anthropic-backed clinical report.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online with Streamlit Community Cloud

This repo is set up for Streamlit Community Cloud:

- Entrypoint: `app.py`
- Dependencies: `requirements.txt`
- Config: `.streamlit/config.toml`

Deploy steps:

1. Push this repo to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click `Create app`.
4. Select this repository and branch.
5. Set the main file path to `app.py`.
6. In `Advanced settings`, choose Python `3.12`.
7. Optional: add this secret if you want AI narration:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

If you skip the secret, the app still works and falls back to a rule-based report.

## Notes

- The `RETOUCH weights path` field is optional. If you don't provide weights, the segmentation model uses random initialization, which is fine for demo mode but not for real clinical-quality output.
- Demo mode works without uploads and is the fastest way to verify a cloud deployment.
