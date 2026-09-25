# run_tests.py
# Batch test of the ICD-10 agent on a fixed set of notes with known expected codes.
# Run:  python run_tests.py          (takes ~15 minutes because of the Groq rate limit)
# Output: test_results.csv (one row per test) + test_results.json (full agent reports)

import csv
import json
import time

import pandas as pd

from agent import analyze_note

WAIT_BETWEEN_TESTS = 60   # seconds; keeps us under Groq's 8,000 tokens/minute free limit

full_note_797 = pd.read_csv("data/selected_labeled_notes.csv").query("`Unnamed: 0` == 797")["combined_text"].iloc[0]

# expected = list of acceptable code PREFIXES (no dots). [] means "no code should be suggested".
TESTS = [
    {"name": "sigmoidoscopy_short", "note": "Flexible sigmoidoscopy. Sigmoid and left colon diverticulosis. Rectal bleeding.", "expected": ["K5731"]},
    {"name": "sigmoidoscopy_full_797", "note": full_note_797, "expected": ["K5731"]},
    {"name": "knee_meniscus", "note": "Right knee pain for 3 weeks. MRI shows a tear of the medial meniscus of the right knee.", "expected": ["S83241", "S8320", "S8323", "S8324"]},
    {"name": "type2_diabetes", "note": "Type 2 diabetes mellitus without complications. Continue metformin.", "expected": ["E119"]},
    {"name": "hypertension", "note": "Essential hypertension, well controlled on lisinopril.", "expected": ["I10"]},
    {"name": "acute_bronchitis", "note": "Cough for 5 days. Diagnosis: acute bronchitis.", "expected": ["J20"]},
    {"name": "ankle_sprain_left", "note": "Twisted left ankle today while running. Exam: sprain of the left ankle. Initial visit.", "expected": ["S9340"]},
    {"name": "uti", "note": "Burning on urination. Urinalysis positive. Diagnosis: urinary tract infection.", "expected": ["N390"]},
    {"name": "chest_pain", "note": "Patient presents with chest pain, cause not yet determined.", "expected": ["R07"]},
    {"name": "dental_caries", "note": "Dental caries on the smooth surface of tooth 19, penetrating into dentin.", "expected": ["K0262"]},
    {"name": "nonsense_greeting", "note": "hello how are you", "expected": []},
    {"name": "nonsense_weather", "note": "The weather is nice today and I went for a walk.", "expected": []},
]


def score(test, report):
    good_codes = [c["code"] for c in report["code_checks"] if c["valid"] and c["from_tools"]]
    if not test["expected"]:
        return ("PASS" if not report["code_checks"] else "FAIL"), good_codes
    hit = any(code.startswith(prefix) for code in good_codes for prefix in test["expected"])
    return ("PASS" if hit else "FAIL"), good_codes


rows, full_reports = [], []
for i, test in enumerate(TESTS, start=1):
    print(f"\n[{i}/{len(TESTS)}] {test['name']}")
    start = time.time()
    try:
        report = analyze_note(test["note"])
        result, good_codes = score(test, report)
        rows.append({
            "test": test["name"],
            "result": result,
            "expected": " / ".join(test["expected"]) or "(no code)",
            "verified_codes_in_answer": " ".join(good_codes),
            "problem_codes": " ".join(report["problem_codes"]),
            "classifier_top1": report["classifier_suggestions"][0]["code"] if report["classifier_suggestions"] else "",
            "classifier_top1_conf": report["classifier_suggestions"][0]["confidence"] if report["classifier_suggestions"] else "",
            "steps": len(report["steps"]),
            "guard_triggered": report["guard_triggered"],
            "seconds": round(time.time() - start, 1),
        })
        full_reports.append({"test": test["name"], "report": report})
    except Exception as error:
        rows.append({"test": test["name"], "result": "ERROR", "expected": " / ".join(test["expected"]),
                     "verified_codes_in_answer": "", "problem_codes": "", "classifier_top1": "",
                     "classifier_top1_conf": "", "steps": "", "guard_triggered": "",
                     "seconds": round(time.time() - start, 1), "error": repr(error)[:300]})
        full_reports.append({"test": test["name"], "error": repr(error)})
    print("  ->", rows[-1]["result"], rows[-1].get("verified_codes_in_answer", ""))

    # save after every test, so nothing is lost if the run stops halfway
    pd.DataFrame(rows).to_csv("test_results.csv", index=False)
    with open("test_results.json", "w") as f:
        json.dump(full_reports, f, indent=2, default=str)

    if i < len(TESTS):
        time.sleep(WAIT_BETWEEN_TESTS)

passed = sum(r["result"] == "PASS" for r in rows)
print(f"\n===== {passed}/{len(rows)} PASSED =====")
print(pd.DataFrame(rows)[["test", "result", "expected", "verified_codes_in_answer", "problem_codes"]].to_string(index=False))
