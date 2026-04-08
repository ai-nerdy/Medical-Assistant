# 🏥 Medical Assistant — RAG-based Healthcare Q&A

An AI-powered medical Q&A application built with **Streamlit** and **Retrieval-Augmented Generation (RAG)**. It answers clinical questions using evidence retrieved from the **Merck Manual**, a comprehensive medical reference with 4,000+ pages across 23 sections.

## Features

- **RAG Pipeline** — Retrieves relevant passages from the Merck Manual and generates grounded, evidence-based answers using GPT-4o-mini.
- **Chat Interface** — Conversational UI with message history.
- **Source Citations** — Responses include section titles and page references.
- **Sample Questions** — Pre-built clinical queries in the sidebar for quick testing.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB (in-memory) |
| Document Loader | PyMuPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

## Project Structure

```
Medical-Assistant/
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── medical_diagnosis_manual.pdf   # Merck Manual PDF (required at runtime)
├── Ankita_Project_2.ipynb         # Original development notebook
├── .gitignore
├── .streamlit/
│   └── config.toml                # Streamlit theme & settings
└── README.md
```

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Medical-Assistant.git
   cd Medical-Assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys** — create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   OPENAI_API_BASE = "https://api.openai.com/v1"
   ```

4. **Place the PDF:**
   Ensure `medical_diagnosis_manual.pdf` is in the project root.

5. **Run:**
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (include the PDF; use [Git LFS](https://git-lfs.github.com/) if it exceeds 100 MB).

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → set main file to `app.py`.

3. In **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   OPENAI_API_BASE = "https://api.openai.com/v1"
   ```

4. Click **Deploy**.

> The first load takes a few minutes while the PDF is indexed and embedded. Subsequent loads use cached data.
