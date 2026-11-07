"""Nyaya Saathi: multilingual AI legal-help assistant for India (Streamlit UI).
All logic lives in core.py. Keep this file in the folder that contains data/vector_store."""
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

import core

load_dotenv()
st.set_page_config(page_title="Nyaya Saathi - AI Legal Helper", page_icon="⚖️", layout="centered")


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


API_KEY = get_api_key()
if not API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to a .env file (local) or to the app's secrets (hosted).")
    st.stop()


@st.cache_resource(show_spinner="Loading the legal database...")
def load_lawyer(api_key: str):
    return core.Lawyer(api_key)


try:
    lawyer = load_lawyer(API_KEY)
except Exception as e:
    st.error(f"Could not open the vector store at `{core.VECTOR_DIR}` (collection `{core.COLLECTION}`). "
             f"Run the ingestion notebook first, or set VECTOR_DIR. Details: {e}")
    st.stop()

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("case", {})
ss.setdefault("last_query", "")
ss.setdefault("language", "English")
ss.setdefault("draft", None)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve", 2, 8, 5)
    min_score = st.slider("Minimum match score", 0.1, 0.6, 0.3, 0.05,
                          help="Below this, the app says it doesn't know instead of guessing.")
    practice = st.toggle("🎭 Practice mode (I play the other side)",
                         help="Rehearse the conversation with your landlord / employer / seller.")
    st.caption(f"LLM: `{core.GROQ_MODEL}`  \nSpeech: `{core.WHISPER_MODEL}`")

    st.header("📁 Your case file")
    if ss.case:
        st.caption("Built only from what you told me. Tell me if something is wrong.")
        st.json(ss.case, expanded=False)
        miss = core.missing_facts(ss.case)
        if miss:
            st.markdown("**Still needed:** " + "; ".join(miss))
        else:
            st.success("Basic facts complete")
    else:
        st.caption("Describe your problem and I will build it here.")

    st.header("📝 Prepare a document")
    kind = st.selectbox("Type", ["Legal notice to the other party", "Complaint to the proper authority"])
    if st.button("Prepare draft", disabled=not ss.case):
        with st.spinner("Drafting..."):
            ss.draft = lawyer.draft(kind, ss.case, ss.last_query or str(ss.case.get("issue", "")),
                                    ss.language, top_k, min_score)
    if st.button("Start a new case"):
        for k in ("messages", "case", "last_query", "draft"):
            ss.pop(k, None)
        st.rerun()

    st.markdown("**Try asking**")
    st.markdown("- मेरी कंपनी ने दो महीने से सैलरी नहीं दी, मैं क्या करूँ?\n"
                "- My landlord is not returning my security deposit.\n"
                "- Amazon did not refund my returned product.")

st.title("⚖️ Nyaya Saathi")
st.caption("Describe your legal problem in any Indian language, by voice or text. "
           "General legal information from Indian Acts. Not legal advice.")


def render_extras(msg: dict):
    if msg.get("needs_lawyer"):
        st.error("⚠️ This looks serious" + (f" ({msg['reason']})" if msg.get("reason") and msg["reason"].lower() != "none" else "")
                 + ". Please speak to a real lawyer soon. Free help: your District Legal Services Authority (NALSA helpline 15100).")
    if msg.get("warning"):
        st.warning(msg["warning"])
    if msg.get("sources"):
        with st.expander("Sources used"):
            for s in msg["sources"]:
                st.markdown(f"**[{s.get('n', '')}] {s['act']}**, page {s['page']} (match {s['score']})")
                st.caption(s["text"][:400] + "...")
    if msg.get("meta"):
        st.caption(msg["meta"])


for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

if ss.draft:
    st.subheader("📄 Draft (review before use)")
    if ss.draft["warning"]:
        st.warning(ss.draft["warning"])
    st.text_area("Draft", ss.draft["text"], height=400, label_visibility="collapsed")
    st.download_button("Download draft", ss.draft["text"], file_name="draft.txt")

audio = st.audio_input("🎤 Speak your problem (Hindi, Tamil, Telugu, Bengali, Marathi, English, ...)")
typed = st.chat_input("...or type your problem here")

question, from_voice = None, False
if audio is not None:
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    fresh = ss.get("last_audio") != digest
    ss.last_audio = digest  # never process the same recording twice
    if fresh and not typed:
        with st.spinner("Listening..."):
            try:
                question, from_voice = lawyer.transcribe(data), True
            except Exception as e:
                st.error(f"Could not transcribe the audio: {e}")
        if from_voice and not question:
            st.warning("I could not hear anything clearly. Please try again or type your problem.")
            question = None
if typed:
    question, from_voice = typed, False

if question:
    shown = f"🎤 {question}" if from_voice else question
    with st.chat_message("user"):
        st.markdown(shown)
    with st.chat_message("assistant"):
        with st.spinner("Searching the law..."):
            try:
                r = lawyer.turn(question, ss.messages, ss.case, top_k, min_score, practice)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(r["answer"])
        msg = {"role": "assistant", "content": r["answer"], "sources": r["hits"], "warning": r["warning"],
               "meta": r["meta"], "needs_lawyer": r["needs_lawyer"], "reason": r["reason"]}
        render_extras(msg)
    ss.messages.append({"role": "user", "content": shown})
    ss.messages.append(msg)
    ss.case, ss.last_query, ss.language = r["case"], r["query"], r["language"]
    st.rerun()  # refresh the sidebar case file

st.divider()
st.caption("For serious matters (court dates, criminal cases, large amounts) please consult a real lawyer. "
           "Free help: your District Legal Services Authority.")