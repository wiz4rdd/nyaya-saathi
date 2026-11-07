# Nyaya Saathi

**Live demo:** https://nyaya-saathi-gzqgf9nll29efnbvhbxdsy.streamlit.app/

Talk (voice or text, in your own language) about a legal problem. It builds a case file, explains your options using
only retrieved passages from Indian Acts, drafts a legal notice or complaint, lets you practise the conversation with
the other side, and tells you when you need a real lawyer. General information, not legal advice.

**Live app:** <paste Streamlit link>

## Covers
Unpaid wages, consumer refunds, security deposits, bounced cheques. Not covered: family law, property disputes,
criminal defence, state-specific rules that are not in the corpus.

## How it avoids invented law
1. Answers only from numbered retrieved passages; every legal statement must cite `[n]`.
2. Code flags section numbers and passage numbers that are not in the retrieved text.
3. Below the match threshold it says it does not know and points to a lawyer or the District Legal Services Authority.
4. Drafts use `[PLACEHOLDERS]` for unknown facts.

## Files
- `app.py`: Streamlit UI
- `core.py`: all RAG and LLM logic (models swappable through environment variables)
- `run_tests.py`: runs 12 test cases and writes unedited results to `raw_outputs/`
- `notebook/pdf_loader.ipynb`: builds the vector store from the Act PDFs
- `data/vector_store/`: the ChromaDB index (collection `legal_docs`)

## Run locally
```
pip install -r requirements.txt
# .env (never commit this file):
# GROQ_API_KEY=...
# GROQ_MODEL=openai/gpt-oss-120b      # swap the answering model here
# FAST_MODEL=llama-3.1-8b-instant     # small model for helper steps
streamlit run app.py
python run_tests.py
```

## Rebuild the index
Put Act PDFs in `data/pdf/`, add each filename and Act name to `ACT_NAMES` in `core.py`, then run the notebook.

## Known limits
Free-tier rate limits slow heavy use. The section check confirms a number appears in the retrieved text, not that it
supports the claim. No state rent laws. Laws may have been amended since the PDFs were downloaded.
