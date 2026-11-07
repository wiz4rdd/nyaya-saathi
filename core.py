"""Nyaya Saathi core: all RAG / LLM logic, no Streamlit. Imported by app.py and run_tests.py."""
import json
import os
import re
import time
from pathlib import Path

BASE = Path(__file__).parent
VECTOR_DIR = os.getenv("VECTOR_DIR", str(BASE / "data" / "vector_store"))
COLLECTION = os.getenv("COLLECTION", "legal_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")      # must match the ingestion notebook
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")     # swap from .env, no code change
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
FAST_MODEL = os.getenv("FAST_MODEL", GROQ_MODEL)  # small model for cheap steps, e.g. llama-3.1-8b-instant

ACT_NAMES = {
    "202504081796627129.pdf": "Legal Services Authorities Act, 1987",
    "A187209.pdf": "Indian Contract Act, 1872",
    "Shop-and-Establishment-Act-Handbook.pdf": "Shops & Establishments handbook (secondary summary, not the Act)",
    "b0620548445580767b5c0d18c95c26f7.pdf": "Code on Social Security, 2020",
    "transferofproperty.pdf": "Transfer of Property Act, 1882",
    "Bharatiya_Nyaya_Sanhita,_2023.pdf": "Bharatiya Nyaya Sanhita, 2023",
    "the-sale-of-goods-act-1930.pdf": "Sale of Goods Act, 1930",
    "negotiable-instrument-act.pdf": "Negotiable Instruments Act, 1881",
    "0e6aea16d11ed1a622f1de43ff3710b4.pdf": "The Model Tenancy Act, 2021",
    "682a44b5426bff2c1f4943ee1b2fd566.pdf": "The Industrial Relations Code, 2020",
}

ANSWER_PROMPT = """You are a legal information assistant for ordinary people in India. You are NOT a lawyer.
Answer ONLY from the numbered passages below.
- After every legal statement add the passage number in square brackets, like [2]. Use ONLY the numbers 1 to {n}. Never state a rule without a number.
- Name an Act or section number only if it appears in a passage. Never cite a section, case, court or time limit that is not in the passages.
- If the passages do not answer the question, say you don't know and suggest a lawyer or the District Legal Services Authority (free legal aid).
- Say clearly what the person's options are and the next 2-4 practical steps, in plain simple words.
- Reply in {language}.
{followup}
- End with one line: this is general information, not legal advice.

The person's case so far (do not add facts that are not here):
{case}

Passages:
{context}

Question: {question}

Answer:"""

ROLEPLAY_PROMPT = """You are helping an ordinary person in India practise a difficult conversation about their legal problem.
Play the OTHER SIDE ({other_party}) realistically but not abusively: give the excuses and objections such a party typically gives.
Rules:
- Stay in character for 2-4 sentences, then add a line starting with "COACH:" giving one tip in {language} for how the person can respond. Base the tip ONLY on the numbered passages and the case; cite the passage number [n]. If the passages give no basis, tell them to ask a lawyer.
- Never invent laws, sections or facts about the case.
- Write the in-character part and the COACH tip in {language}.

Case:
{case}

Passages:
{context}

Conversation so far:
{history}

Person says: {question}

Reply:"""

UNDERSTAND_PROMPT = """You help a legal-information app for India. Read the conversation and the latest user message.
Reply in exactly this format and nothing else:
LANGUAGE: <language the latest user message is written in>
QUERY: <short English search query for a database of Indian laws; use the key facts from the WHOLE conversation: who, what, amounts, how long>
TOPIC: <one of: wages, consumer_refund, rent_deposit, contract, property, criminal, other>
NEEDS_LAWYER: <yes or no>
REASON: <short English phrase, or none>
LEGAL_PROBLEM: <yes if the user describes or asks about a legal problem, no for greetings or small talk>
Answer NEEDS_LAWYER yes if there is an arrest, FIR, police, a court summons or hearing date, violence or threats, a deadline about to expire, or a large sum of money.

Conversation so far:
{history}

Latest user message: {question}"""

CASE_PROMPT = """Maintain a case file for a person describing a legal problem. Update the previous case file using ONLY what the person has said. Use null (or []) for anything not said. Never guess or infer facts.
Keys: issue, other_party, what_happened, key_dates (list of strings), amount_inr, proof_they_have (list), what_they_want.
Reply with JSON only, no other text.

Previous case file: {case}

What the person has said:
{conversation}"""

DRAFT_PROMPT = """Draft a {kind} for an ordinary person in India.
Rules:
- Use ONLY facts in the case file. For every missing fact write a placeholder in capitals, like [DATE OF AGREEMENT]. Never invent names, dates, amounts or addresses.
- Mention an Act or section only if it appears in the passages below. If none applies, do not cite any law; just state the facts and the demand.
- Write the draft in English, formal and short, with a reasonable deadline for response (e.g. 15 days) and what will happen if it is ignored (approach the proper forum).
- After the draft add a heading CHECK BEFORE SENDING with: the placeholders to fill, the proof to attach, and a reminder to have a lawyer or the District Legal Services Authority review it.
- Finally add a 3-line plain summary in {language}.

Case file:
{case}

Passages:
{context}

Draft:"""

NO_ANSWER = ("I could not find this in the legal documents I have, so I don't want to guess. "
             "Please talk to a lawyer or your District Legal Services Authority (free legal aid).")
RISK_RE = re.compile(r"\b(arrest\w*|FIR|police|summons|bail|threat\w*|violen\w*|assault\w*|court notice|hearing|eviction)\b", re.I)
REQUIRED_FACTS = {
    "other_party": "who the other party is",
    "key_dates": "the key dates",
    "amount_inr": "the amount involved",
    "proof_they_have": "what proof they have (messages, receipts, agreement, payslips)",
}


def build_prompt(template: str, **values) -> str:
    for k, v in values.items():  # str.replace, not format: legal text contains braces
        template = template.replace("{" + k + "}", str(v))
    return template


def act_name(source_file: str) -> str:
    return ACT_NAMES.get(source_file, source_file)


def history_text(history, user_only=False, n=8) -> str:
    rows = []
    for m in history[-n:]:
        if user_only and m["role"] != "user":
            continue
        rows.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:600]}")
    return "\n".join(rows) or "(none)"


def parse_json(text: str):
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return None


def missing_facts(case: dict) -> list:
    return [label for key, label in REQUIRED_FACTS.items()
            if not case.get(key) or str(case.get(key)).strip().lower() in ("null", "none", "unknown")]


def unsupported_sections(answer: str, context: str):
    """Section numbers cited in the answer that never appear next to 'section' in the retrieved text."""
    cited = set(re.findall(r"(?:[Ss]ection|[Ss]ec\.|धारा)\s*(\d+[A-Za-z]?)", answer))
    ctx_secs = set(re.findall(r"(?:[Ss]ection|[Ss]ec\.|धारा)s?\s*(\d+[A-Za-z]?)", context))
    ctx_heads = set(re.findall(r"(?m)^\s*(\d+[A-Za-z]?)\.\s", context))  # "17. Time limit for payment..."
    ctx_inline = set(re.findall(r"[\.\]\)]\s+(\d+[A-Za-z]?)\.\s+[A-Z(]", context))
    known = ctx_secs | ctx_heads | ctx_inline
    return sorted(s for s in cited if s not in known)


def check_citations(answer: str, n_hits: int):
    cited = sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", answer)})
    return [c for c in cited if 1 <= c <= n_hits], [c for c in cited if c < 1 or c > n_hits]


class Lawyer:
    def __init__(self, api_key: str):
        import chromadb
        from groq import Groq
        from langchain_groq import ChatGroq
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.collection = chromadb.PersistentClient(path=VECTOR_DIR).get_collection(COLLECTION)
        self.llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL, temperature=0.1, max_tokens=2048)
        self.fast_llm = (self.llm if FAST_MODEL == GROQ_MODEL else
                         ChatGroq(groq_api_key=api_key, model_name=FAST_MODEL, temperature=0.0, max_tokens=1024))
        self.groq = Groq(api_key=api_key)

    def _ask(self, prompt: str, fast: bool = False) -> str:
        """Call the LLM; on a Groq rate limit (429) wait the time it asks for and retry."""
        llm = self.fast_llm if fast else self.llm
        for attempt in range(8):
            try:
                return llm.invoke(prompt).content
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "rate limit" not in msg.lower():
                    raise
                m = re.search(r"try again in ([\d.]+)s", msg)
                time.sleep((float(m.group(1)) + 1) if m else 5 * (attempt + 1))
        raise RuntimeError("Still rate-limited after several retries. Wait a minute or use a smaller model.")

    # ---- speech
    def transcribe(self, audio_bytes: bytes) -> str:
        r = self.groq.audio.transcriptions.create(file=("question.wav", audio_bytes), model=WHISPER_MODEL, temperature=0.0)
        return r.text.strip()

    # ---- understanding
    def understand(self, question: str, hist: str) -> dict:
        out = {"language": "the same language as the question", "query": question,
               "topic": "other", "needs_lawyer": False, "reason": "", "legal": True}
        try:
            raw = self._ask(build_prompt(UNDERSTAND_PROMPT, history=hist, question=question), fast=True)
            for key, pat in (("language", r"LANGUAGE:\s*(.+)"), ("query", r"QUERY:\s*(.+)"),
                             ("topic", r"TOPIC:\s*(\w+)"), ("reason", r"REASON:\s*(.+)")):
                m = re.search(pat, raw)
                if m:
                    out[key] = m.group(1).strip()
            m = re.search(r"LEGAL_PROBLEM:\s*(\w+)", raw)
            out["legal"] = not (m and m.group(1).lower().startswith("n"))
            m = re.search(r"NEEDS_LAWYER:\s*(\w+)", raw)
            out["needs_lawyer"] = bool(m and m.group(1).lower().startswith("y"))
        except Exception:
            pass
        if not out["needs_lawyer"] and RISK_RE.search(out["query"]):  # rule-based backup
            out["needs_lawyer"], out["reason"] = True, out["reason"] or "police/court/threat keywords"
        return out

    def extract_case(self, history, question: str, case: dict) -> dict:
        convo = history_text(history, user_only=True, n=20) + f"\nUser: {question}"
        try:
            data = parse_json(self._ask(build_prompt(CASE_PROMPT, case=json.dumps(case), conversation=convo), fast=True))
            return data if isinstance(data, dict) else case
        except Exception:
            return case

    # ---- retrieval
    def retrieve(self, query: str, top_k: int, min_score: float):
        emb = self.embedder.encode([query])[0].tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1 - dist  # cosine distance
            if score >= min_score:
                src = meta.get("source_file", "unknown")
                hits.append({"text": doc, "source": src, "act": act_name(src),
                             "page": meta.get("page_label", meta.get("page", "?")), "score": round(score, 2)})
        return hits

    def say_in(self, language: str, text: str) -> str:
        if language.lower().startswith("english"):
            return text
        try:
            return self._ask(f"Translate into {language}. Output only the translation.\n\n{text}")
        except Exception:
            return text


    def turn(self, question, history, case, top_k=5, min_score=0.3, practice=False) -> dict:
        u = self.understand(question, history_text(history))
        if not u["legal"] and not case:
            greet = ("Hello! I'm Nyaya Saathi. Tell me what happened, for example unpaid salary, a refund that never "
                     "came, or a deposit your landlord won't return, and I'll help you understand your options.")
            return {"case": {}, "missing": [], "language": u["language"], "query": "", "needs_lawyer": False,
                    "reason": "", "topic": "other", "answer": self.say_in(u["language"], greet), "hits": [],
                    "warning": "", "meta": f"{u['language']} | greeting, no search"}
        case = self.extract_case(history, question, case or {})
        missing = missing_facts(case)
        hits = self.retrieve(u["query"], top_k, min_score)
        base = {"case": case, "missing": missing, "language": u["language"], "query": u["query"],
                "needs_lawyer": u["needs_lawyer"], "reason": u["reason"], "topic": u["topic"]}
        if not hits:
            return {**base, "answer": self.say_in(u["language"], NO_ANSWER), "hits": [], "warning": "",
                    "meta": f"{u['language']} | no passage above threshold"}
        context = "\n\n".join(f"[{i}] {h['act']}, page {h['page']}\n{h['text']}" for i, h in enumerate(hits, 1))
        case_json = json.dumps(case, ensure_ascii=False)
        if practice:
            prompt = build_prompt(ROLEPLAY_PROMPT, other_party=case.get("other_party") or "the other party",
                                  language=u["language"], case=case_json, context=context,
                                  history=history_text(history), question=question)
        else:
            followup = (f"- At the end, ask ONE short question to learn: {missing[0]}." if missing else "")
            prompt = build_prompt(ANSWER_PROMPT, n=len(hits), language=u["language"], followup=followup, case=case_json,
                                  context=context, question=question)
        answer = self._ask(prompt)
        valid, invalid = check_citations(answer, len(hits))
        bad = unsupported_sections(answer, context)
        warns = []
        if bad:
            warns.append(f"Section(s) {', '.join(bad)} were not found in the retrieved text. Please verify before relying on them.")
        if invalid:
            warns.append(f"The answer points to passage(s) {invalid} that do not exist.")
        if not valid and not practice:
            warns.append("The answer has no source markers, so treat it with extra caution.")
        shown = [dict(hits[i - 1], n=i) for i in valid] or [dict(h, n=i) for i, h in enumerate(hits, 1)]
        return {**base, "answer": answer, "hits": shown, "warning": " ".join(warns),
                "meta": f"{u['language']} | topic: {u['topic']} | searched: {u['query']}"}

    # ---- documents
    def draft(self, kind: str, case: dict, query: str, language: str, top_k=5, min_score=0.3) -> dict:
        hits = self.retrieve(query, top_k, min_score)
        context = "\n\n".join(f"[{i}] {h['act']}, page {h['page']}\n{h['text']}" for i, h in enumerate(hits, 1)) or "(no relevant passage found)"
        text = self._ask(build_prompt(DRAFT_PROMPT, kind=kind, language=language,
                                      case=json.dumps(case, ensure_ascii=False), context=context))
        bad = unsupported_sections(text, context)
        warning = f"Section(s) {', '.join(bad)} were not found in the retrieved text. Remove or verify them." if bad else ""
        return {"text": text, "warning": warning, "hits": hits}
