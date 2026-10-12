"""
End-to-end submission pipeline for WTQ 2026.

Reads the official test templates from test_data/test/csv/,
processes every bill image under test_data/test/bills/,
generates output/level1.csv and output/level2.csv,
validates both, then packages your-submission.zip.

Usage:
    .venv/Scripts/python.exe scripts/generate_submission.py [--refresh] [--bills B1 B2 ...]

Options:
    --refresh    Force re-extraction (bypass cache)
    --bills      Only process the listed bill_ids (space-separated)
    --no-zip     Skip ZIP packaging (CSV generation + validation only)
    --validate-only  Only validate existing CSVs
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import zipfile
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

# Ensure bill_decoder is importable from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bill_decoder.config import Settings
from bill_decoder.engine import atomic_write, ExtractionProblem
from bill_decoder.images import ImageOptions
from bill_decoder.qa import BillQA, Question
from bill_decoder.schema import dumps, parse_json
from bill_decoder.api import Gemini, APIProblem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BILLS_DIR = ROOT / "test_data" / "test" / "bills"
TEMPLATE_L1 = ROOT / "test_data" / "test" / "csv" / "level1.csv"
TEMPLATE_L2 = ROOT / "test_data" / "test" / "csv" / "level2.csv"
OUT_DIR = ROOT / "output"
OUT_L1 = OUT_DIR / "level1.csv"
OUT_L2 = OUT_DIR / "level2.csv"
CACHE_L1 = ROOT / ".cache" / "level1"
CACHE_L2 = ROOT / ".cache" / "level2"
ZIP_PATH = ROOT / "your-submission.zip"

REQUIRED_L1_FIELDS = [
    "provider", "tariff", "sanctioned_load_kw", "bill_month",
    "reading_date", "issue_date", "due_date", "previous_reading",
    "current_reading", "units_consumed", "charges", "total_charges",
    "taxes", "total_taxes", "current_bill", "arrears",
    "payable_within_due_date", "payable_after_due_date",
]

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


# ---------------------------------------------------------------------------
# Template reading
# ---------------------------------------------------------------------------

def read_l1_template():
    """Returns list of bill_ids in order."""
    with TEMPLATE_L1.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == ["bill_id", "json"], \
            f"Unexpected level1 headers: {reader.fieldnames}"
        return [row["bill_id"] for row in reader]


def read_l2_template():
    """Returns list of dicts with bill_id, question_id, question in template order."""
    with TEMPLATE_L2.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == ["bill_id", "question_id", "question", "answer"], \
            f"Unexpected level2 headers: {reader.fieldnames}"
        return [
            {"bill_id": row["bill_id"], "question_id": row["question_id"], "question": row["question"]}
            for row in reader
        ]


# ---------------------------------------------------------------------------
# Bill image finder
# ---------------------------------------------------------------------------

def find_image(bill_id: str):
    """Find the image file for a given bill_id regardless of extension."""
    for ext in IMAGE_EXTS:
        candidate = BILLS_DIR / (bill_id + ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Extraction: Level 1
# ---------------------------------------------------------------------------

def extract_bill(bill_id: str, api: Gemini, refresh: bool):
    """Extract Level 1 JSON for one bill. Returns the bill dict or None on failure."""
    from bill_decoder.engine import extract
    img = find_image(bill_id)
    if img is None:
        print(f"  [MISSING IMAGE] {bill_id}", flush=True)
        return None
    try:
        opts = ImageOptions(detail_crops=True)
        envelope = extract(img, api, CACHE_L1, opts, refresh)
        bill = envelope["extraction"]["bill"]
        key_prefix = envelope.get("cache_key", "")[:8]
        print(f"  [L1 OK] {bill_id}  key={key_prefix}", flush=True)
        return bill
    except (ExtractionProblem, APIProblem, ValueError, OSError) as exc:
        print(f"  [L1 FAIL] {bill_id}: {type(exc).__name__}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Q&A: Level 2
# ---------------------------------------------------------------------------

def answer_bill(bill_id: str, questions: list, service: BillQA, refresh: bool):
    """Answer all questions for a bill. Returns {question_id: answer} or None."""
    import re
    img = find_image(bill_id)
    if img is None:
        print(f"  [MISSING IMAGE] {bill_id}", flush=True)
        return None
    try:
        needs_focus = any(
            re.search(r'\b(slab|rate per|per.unit rate|pay(?:ment)? on|payment band|paid on|fpa|fca|fuel)\b',
                      q.question, re.I)
            for q in questions
        )
        report = service.answer(img, questions, refresh=refresh, focus_evidence=needs_focus)
        result = {r["question_id"]: r["answer"] for r in report["results"]}
        hit = report.get("answer_cache_hit", False)
        calls = report.get("api_calls", 0)
        print(f"  [L2 OK] {bill_id}  answered={len(result)}  cache_hit={hit}  api_calls={calls}", flush=True)
        return result
    except (ExtractionProblem, APIProblem, ValueError, OSError) as exc:
        print(f"  [L2 FAIL] {bill_id}: {type(exc).__name__}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_level1_csv(bill_ids: list, results: dict):
    """Write output/level1.csv with compact single-line JSON per row."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_L1.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bill_id", "json"])
        for bill_id in bill_ids:
            bill = results.get(bill_id)
            json_str = dumps(bill) if bill is not None else ""
            writer.writerow([bill_id, json_str])
    tmp.replace(OUT_L1)
    print(f"\n[CSV] Written: {OUT_L1} ({OUT_L1.stat().st_size} bytes)", flush=True)


def write_level2_csv(template_rows: list, answers: dict):
    """Write output/level2.csv preserving exact template order."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_L2.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bill_id", "question_id", "question", "answer"])
        for tmpl in template_rows:
            bid = tmpl["bill_id"]
            qid = tmpl["question_id"]
            bill_answers = answers.get(bid)
            answer = bill_answers.get(qid, "") if bill_answers else ""
            writer.writerow([bid, qid, tmpl["question"], answer])
    tmp.replace(OUT_L2)
    print(f"[CSV] Written: {OUT_L2} ({OUT_L2.stat().st_size} bytes)", flush=True)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_level1(bill_ids: list) -> list:
    issues = []
    if not OUT_L1.exists():
        return [f"MISSING: {OUT_L1}"]
    with OUT_L1.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        if headers != ["bill_id", "json"]:
            issues.append(f"L1 header wrong: {headers}")
        rows = list(reader)

    found_ids = [r["bill_id"] for r in rows]
    if found_ids != bill_ids:
        issues.append(f"L1 bill_id order/content mismatch.\n  Expected: {bill_ids}\n  Got:      {found_ids}")

    for row in rows:
        bid = row["bill_id"]
        cell = row.get("json", "")
        if not cell:
            issues.append(f"L1 empty json for {bid}")
            continue
        try:
            data = json.loads(cell, parse_float=Decimal, parse_int=Decimal)
        except json.JSONDecodeError as exc:
            issues.append(f"L1 JSON parse error {bid}: {exc}")
            continue
        for field in REQUIRED_L1_FIELDS:
            if field not in data:
                issues.append(f"L1 missing field '{field}' in {bid}")
        for fname in ["charges", "taxes"]:
            v = data.get(fname)
            if not isinstance(v, list):
                issues.append(f"L1 {bid}.{fname} must be list, got {type(v).__name__}")
        for fname in ["sanctioned_load_kw", "previous_reading", "current_reading",
                      "units_consumed", "total_charges", "total_taxes",
                      "current_bill", "arrears", "payable_within_due_date", "payable_after_due_date"]:
            v = data.get(fname)
            if isinstance(v, str):
                issues.append(f"L1 {bid}.{fname} must be number or null, not string")
    return issues


def validate_level2(template_rows: list) -> list:
    issues = []
    if not OUT_L2.exists():
        return [f"MISSING: {OUT_L2}"]
    with OUT_L2.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        if headers != ["bill_id", "question_id", "question", "answer"]:
            issues.append(f"L2 header wrong: {headers}")
        rows = list(reader)

    if len(rows) != len(template_rows):
        issues.append(f"L2 row count: expected {len(template_rows)}, got {len(rows)}")

    for i, (actual, tmpl) in enumerate(zip(rows, template_rows), 1):
        for key in ["bill_id", "question_id", "question"]:
            if actual.get(key) != tmpl[key]:
                issues.append(f"L2 row {i} {key}: expected '{tmpl[key]}', got '{actual.get(key)}'")
        if not actual.get("answer", "").strip():
            bid = actual.get("bill_id", "?")
            qid = actual.get("question_id", "?")
            issues.append(f"L2 blank answer at row {i} ({bid}/{qid})")

    return issues


# ---------------------------------------------------------------------------
# ZIP packager
# ---------------------------------------------------------------------------

def build_zip():
    """Package output/ and source/ into your-submission.zip."""
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Output CSVs
        for csv_path in [OUT_L1, OUT_L2]:
            if csv_path.exists():
                zf.write(csv_path, f"output/{csv_path.name}")
            else:
                print(f"  [ZIP WARN] Missing: {csv_path}", flush=True)

        # bill_decoder package
        src = ROOT / "bill_decoder"
        for fp in sorted(src.rglob("*")):
            if fp.is_file() and "__pycache__" not in fp.parts:
                zf.write(fp, "source/" + str(fp.relative_to(ROOT)))

        # Scripts (exclude __pycache__)
        scripts_src = ROOT / "scripts"
        for fp in sorted(scripts_src.rglob("*.py")):
            if fp.is_file():
                zf.write(fp, "source/" + str(fp.relative_to(ROOT)))

        # Config and documentation
        for name in ["requirements.txt", ".env.example", "README.md"]:
            fp = ROOT / name
            if fp.exists():
                zf.write(fp, f"source/{name}")

    size = ZIP_PATH.stat().st_size
    print(f"\n[ZIP] Written: {ZIP_PATH} ({size:,} bytes = {size/1024/1024:.2f} MB)", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"  {info.file_size:>10,}  {name}", flush=True)
    if size > 15 * 1024 * 1024:
        print(f"[ZIP WARN] Size {size/1024/1024:.2f} MB exceeds 15 MB limit!", flush=True)
    return size


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WTQ 2026 submission pipeline")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--bills", nargs="+", metavar="BILL_ID")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    print("\n=== READING OFFICIAL TEMPLATES ===", flush=True)
    bill_ids = read_l1_template()
    template_rows = read_l2_template()
    unique_bills_l2 = len(set(r["bill_id"] for r in template_rows))
    unique_qs_l2 = len(set(r["question_id"] for r in template_rows))
    print(f"  Level 1: {len(bill_ids)} bills: {bill_ids}", flush=True)
    print(f"  Level 2: {len(template_rows)} rows ({unique_bills_l2} bills x {unique_qs_l2} questions)", flush=True)

    if args.validate_only:
        print("\n=== VALIDATION ONLY ===", flush=True)
        l1_issues = validate_level1(bill_ids)
        l2_issues = validate_level2(template_rows)
        _print_validation(l1_issues, l2_issues)
        return 0 if not l1_issues and not l2_issues else 1

    process_ids_set = set(args.bills) if args.bills else set(bill_ids)

    # --- Level 1 ---
    print("\n=== LEVEL 1 EXTRACTION ===", flush=True)
    settings = Settings.load()
    api = Gemini(settings)
    l1_results: dict = {}
    permanent_failure = None

    try:
        for bill_id in bill_ids:
            if bill_id not in process_ids_set:
                l1_results[bill_id] = None
                continue
            if permanent_failure:
                print(f"  [L1 SKIP] {bill_id}: permanent API failure earlier", flush=True)
                l1_results[bill_id] = None
                continue
            result = extract_bill(bill_id, api, args.refresh)
            l1_results[bill_id] = result
            if result is None:
                # Check if this was a permanent failure (auth/model)
                # extract_bill already reported; continue to next bill
                pass
    finally:
        api.close()

    print("\n=== WRITING LEVEL 1 CSV ===", flush=True)
    write_level1_csv(bill_ids, l1_results)

    # --- Level 2 ---
    print("\n=== LEVEL 2 QUESTION ANSWERING ===", flush=True)
    l2_answers: dict = {}

    questions_by_bill: dict = {}
    for row in template_rows:
        bid = row["bill_id"]
        if bid not in questions_by_bill:
            questions_by_bill[bid] = []
        questions_by_bill[bid].append(Question(question_id=row["question_id"], question=row["question"]))

    with BillQA(settings, cache_dir=CACHE_L2, level1_cache=CACHE_L1) as service:
        for bill_id in bill_ids:
            if bill_id not in process_ids_set:
                l2_answers[bill_id] = None
                continue
            questions = questions_by_bill.get(bill_id, [])
            if not questions:
                print(f"  [L2 SKIP] {bill_id}: no questions", flush=True)
                l2_answers[bill_id] = {}
                continue
            l2_answers[bill_id] = answer_bill(bill_id, questions, service, args.refresh)

    print("\n=== WRITING LEVEL 2 CSV ===", flush=True)
    write_level2_csv(template_rows, l2_answers)

    print("\n=== VALIDATION ===", flush=True)
    l1_issues = validate_level1(bill_ids)
    l2_issues = validate_level2(template_rows)
    _print_validation(l1_issues, l2_issues)

    if not args.no_zip and not l1_issues and not l2_issues:
        print("\n=== BUILDING SUBMISSION ZIP ===", flush=True)
        build_zip()
    elif args.no_zip:
        print("\n[SKIP] ZIP packaging skipped (--no-zip)", flush=True)
    else:
        print("\n[SKIP] ZIP packaging skipped due to validation failures", flush=True)

    elapsed = time.time() - t_start
    l1_ok = sum(1 for v in l1_results.values() if v is not None)
    l2_ok = sum(1 for v in l2_answers.values() if v is not None)
    total_answered = sum(len(v) for v in l2_answers.values() if v)

    zip_info = str(ZIP_PATH) if ZIP_PATH.exists() else "NOT BUILT"

    print(f"""
=== FINAL REPORT ===
  Official test bills:        {len(bill_ids)}
  Bills successfully L1:      {l1_ok} / {len(bill_ids)}
  Bills successfully L2:      {l2_ok} / {len(bill_ids)}
  Total questions answered:   {total_answered} / {len(template_rows)}
  Level 1 CSV:                {OUT_L1}
  Level 2 CSV:                {OUT_L2}
  L1 validation:              {"PASS" if not l1_issues else f"FAIL ({len(l1_issues)} issues)"}
  L2 validation:              {"PASS" if not l2_issues else f"FAIL ({len(l2_issues)} issues)"}
  ZIP:                        {zip_info}
  Elapsed:                    {elapsed:.1f}s
  Ready to upload:            {"YES" if not l1_issues and not l2_issues and ZIP_PATH.exists() else "NO -- see issues above"}
""", flush=True)

    return 0 if not l1_issues and not l2_issues else 1


def _print_validation(l1_issues, l2_issues):
    if l1_issues:
        print(f"  Level 1: FAIL ({len(l1_issues)} issues)", flush=True)
        for issue in l1_issues:
            print(f"    - {issue}", flush=True)
    else:
        print("  Level 1: PASS", flush=True)

    if l2_issues:
        print(f"  Level 2: FAIL ({len(l2_issues)} issues)", flush=True)
        for issue in l2_issues:
            print(f"    - {issue}", flush=True)
    else:
        print("  Level 2: PASS", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
