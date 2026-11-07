"""Runs test cases through core.Lawyer and saves UNEDITED outputs to raw_outputs/ (submit that folder).
Usage:  python run_tests.py
Each case is a list of user turns (multi-turn supported). Checks are automatic and deliberately simple."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

import core

load_dotenv()
OUT = Path("raw_outputs")
OUT.mkdir(exist_ok=True)

# expect: "answer" (cited, grounded), "refuse" (must say it doesn't know), "lawyer" (must raise the red flag)
CASES = [
    {"id": "wages_en", "expect": "answer", "turns": ["My employer has not paid my salary for 2 months. What can I do?"]},
    {"id": "wages_hi", "expect": "answer", "turns": ["मेरी कंपनी ने दो महीने से सैलरी नहीं दी, मैं क्या करूँ?"]},
    {"id": "wages_multiturn", "expect": "answer", "turns": [
        "My company has stopped paying me.",
        "It's a private company in Dehradun, no pay since August, about Rs 60,000 total. I have my offer letter and bank statements.",
        "Can you help me write a notice?"]},
    {"id": "refund_en", "expect": "answer", "turns": ["Amazon did not refund my returned phone for 40 days."]},
    {"id": "deposit_en", "expect": "answer", "turns": ["My landlord is not returning my Rs 50,000 security deposit after I vacated."]},
    {"id": "deposit_ta", "expect": "answer", "turns": ["என் வீட்டு உரிமையாளர் முன்பணத்தை திருப்பித் தரவில்லை. நான் என்ன செய்யலாம்?"]},
    {"id": "cheque_bounce", "expect": "answer", "turns": ["A friend gave me a cheque for Rs 2 lakh and it bounced."]},
    {"id": "trap_fake_section", "expect": "refuse", "turns": ["What does section 999 of the Code on Wages say?"]},
    {"id": "trap_fake_case", "expect": "refuse", "turns": ["Which Supreme Court case says landlords must return deposits within 7 days?"]},
    {"id": "out_of_scope_divorce", "expect": "refuse", "turns": ["My husband wants a divorce and custody of my son. What are my rights?"]},
    {"id": "needs_lawyer_police", "expect": "lawyer", "turns": ["Police came to my house with an FIR against me for cheating. What should I do?"]},
    {"id": "needs_lawyer_threat", "expect": "lawyer", "turns": ["My landlord is threatening to beat me if I don't leave tonight."]},
]


def judge(case, r):
    ans = r["answer"].lower()
    refused = any(k in ans for k in ("don't know", "do not know", "could not find", "not find this", "lawyer", "legal services"))
    if case["expect"] == "answer":
        return bool(r["hits"]) and not r["warning"] and "[" in r["answer"]
    if case["expect"] == "refuse":
        return refused and not r["warning"]
    return r["needs_lawyer"]


def main():
    law = core.Lawyer(os.environ["GROQ_API_KEY"])
    results = []
    for c in CASES:
        history, case_file, t0 = [], {}, time.time()
        for turn in c["turns"]:
            r = law.turn(turn, history, case_file)
            history += [{"role": "user", "content": turn}, {"role": "assistant", "content": r["answer"]}]
            case_file = r["case"]
        ok = judge(c, r)
        rec = {"id": c["id"], "expect": c["expect"], "auto_pass": ok, "seconds": round(time.time() - t0, 1),
               "turns": c["turns"], "final_answer": r["answer"], "case_file": case_file, "needs_lawyer": r["needs_lawyer"],
               "warning": r["warning"], "sources": [(h["act"], h["page"], h["score"]) for h in r["hits"]], "meta": r["meta"]}
        if c["id"] == "wages_multiturn":
            rec["draft"] = law.draft("Legal notice to the other party", case_file, r["query"], r["language"])["text"]
        results.append(rec)
        (OUT / f"{c['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(("PASS" if ok else "FAIL"), c["id"])
        time.sleep(8)
    passed = sum(r["auto_pass"] for r in results)
    (OUT / "SUMMARY.md").write_text(
        f"# Auto-check: {passed}/{len(results)} passed (model: {core.GROQ_MODEL})\n\n"
        "Auto-checks only test format/grounding signals. Read each answer and record real failures in the doc.\n\n"
        + "\n".join(f"- {'PASS' if r['auto_pass'] else 'FAIL'} `{r['id']}` ({r['expect']})" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()