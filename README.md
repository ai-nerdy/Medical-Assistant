# Medical Knowledge Assistant — Deployment
# Medical Knowledge Assistant (Project2)

A Streamlit-based Retrieval-Augmented Generation (RAG) app that lets you upload or point to a medical reference PDF (for example, the Merck Manual), builds an embeddings-backed knowledge base, and answers clinical questions grounded in the source material.

**Key features**
- Upload or use an existing PDF to create a local knowledge base
- Vector embeddings with ChromaDB + OpenAI embeddings
- Retrieval + generation pipeline returning evidence-based answers with source citations
- Sidebar configuration (API key, model params) and conversation history

**Important:** This project was developed and tested with Python 3.13. Using Python 3.14 on Windows may cause wheel/build problems for some dependencies (for example, Pillow). If you see build errors while installing dependencies, recreate the venv with Python 3.13.

---

**Quick Start — Windows**

1. Install Python 3.13 (recommended).
2. From the repository root create a dedicated venv and install requirements:

```powershell
& "C:\\Path\\To\\Python313\\python.exe" -m venv .venv313
& ".\.venv313\\Scripts\\Activate.ps1"
python -m pip install --upgrade pip
pip install -r Project2/requirements.txt
```

3. Run the app (with the venv active):

```powershell
& ".\.venv313\\Scripts\\python.exe" -m streamlit run Project2/app.py
```

Open the Local URL printed by Streamlit (e.g., http://localhost:8501 or :8502).

---

**Configuration / API keys**

You must provide an OpenAI API key. Options:

- Sidebar input (quick but less secure)
- `Project2/config.json` next to `app.py`:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_API_BASE": "https://api.openai.com/v1"  
}
```

- Streamlit secrets (`.streamlit/secrets.toml`):

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_API_BASE = "https://api.openai.com/v1"  # optional
```

- Environment variables (then restart terminal/Streamlit):

```powershell
setx OPENAI_API_KEY "sk-..."
setx OPENAI_API_BASE "https://api.openai.com/v1"
```

---

**Provide a PDF**

- Use `Use existing PDF file` in the sidebar and supply either a workspace-relative path or the absolute path. Examples:
  - Relative: `Project2/medical_diagnosis_manual.pdf`
  - Absolute: `C:\\Users\\avsan\\OneDrive\\Great Learning Gen AI\\Project_2\\Project2\\medical_diagnosis_manual.pdf`

- Or choose `Upload new PDF` and click `Process PDF` to upload and build the knowledge base for the session.

The app attempts multiple path resolutions automatically. If you see "File not found", verify the path and try an absolute path.

---

**File map**
- `Project2/app.py` — main Streamlit application
- `Project2/requirements.txt` — dependencies
- `Project2/medical_diagnosis_manual.pdf` — example PDF included

---

**Developer notes & known issues**
- The repo includes code to handle LangChain import variations and retriever API differences; if you update LangChain, watch for breaking API changes.
- Pillow and some binaries may not ship wheels for Python 3.14 on Windows; use Python 3.13 if you encounter build failures.
- Vectorstore is in-memory by default; for production, persist Chroma to disk or a managed store.

---

**Docker**

Build and run the container (example):

```bash
# build from repository root
docker build -t medical-assistant:latest Project2

# run (pass your API key)
docker run -e OPENAI_API_KEY=sk-... -p 8501:8501 medical-assistant:latest
```

---

**Troubleshooting**
- `ModuleNotFoundError`: make sure you installed `requirements.txt` into the same Python used to run Streamlit.
- `Error loading credentials: No secrets found`: add API key via sidebar, `config.json`, environment variables, or `.streamlit/secrets.toml`.
- `VectorStoreRetriever has no get_relevant_documents`: ensure PDF processing completed and the vectorstore was created (sidebar should show `System Ready`). The app contains retriever fallbacks to handle multiple langchain/chroma versions.

---

If you want, I can also add:
- `CONTRIBUTING.md` with dev flow and testing steps
- `.env.example` and safer local dev secret handling
- a persistent Chroma example

License: add your preferred license file if you plan to publish this project.
