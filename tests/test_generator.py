"""
ทดสอบโมดูล 1 (data_generator/generate.py)
ตรวจว่าสร้างไฟล์ครบ จำนวนปัญหาในไฟล์ตรงกับไฟล์เฉลย และรันซ้ำได้ข้อมูลเหมือนเดิม
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_generator"))
import generate  # noqa: E402

RAW_FILES = ["nimman_sales.csv", "cmu_sales.xlsx", "thaphae_sales.json",
             "members.csv", "products.csv", "holidays.csv"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    root = tmp_path_factory.mktemp("run1")
    result = generate.main(out_root=root, verbose=False)
    return root, result["summary"]


def load_thaphae(root):
    return json.loads((root / "data/raw/thaphae_sales.json").read_text(encoding="utf-8"))["receipts"]


def test_all_files_created(out):
    root, _ = out
    for name in RAW_FILES:
        assert (root / "data/raw" / name).exists(), name
    assert (root / "data/generator_log/injected_issues.json").exists()
    assert (root / "docs/data_issues.md").exists()


def test_nimman_duplicate_rows_match_log(out):
    root, summary = out
    nm = pd.read_csv(root / "data/raw/nimman_sales.csv", dtype=str)
    assert nm.duplicated().sum() >= summary["B1"]


def test_thaphae_duplicate_bills_match_log(out):
    root, summary = out
    ids = pd.Series([r["receipt_id"] for r in load_thaphae(root)])
    assert ids.duplicated().sum() == summary["B2"]


def test_thaphae_offline_days_have_no_data(out):
    root, _ = out
    # timestamp เป็น UTC ต้องบวก 7 ชั่วโมงก่อนเทียบกับวันที่ไทย
    local_days = {(pd.Timestamp(r["timestamp"][:-1]) + pd.Timedelta(hours=7)).date() for r in load_thaphae(root)}
    for d in generate.THAPHAE_OFFLINE_DAYS:
        assert d not in local_days


def test_members_count_and_missing(out):
    root, summary = out
    m = pd.read_csv(root / "data/raw/members.csv")
    assert len(m) == generate.N_MEMBERS + summary["B3"]
    # นับเฉพาะสมาชิกตัวจริง 800 คนแรก เพราะแถวที่สมัครซ้ำ (B3) คัดลอกค่าว่างมาจากแถวเดิมด้วย
    original = m.iloc[:generate.N_MEMBERS]
    assert original["birth_year"].isna().sum() == summary["B5"]
    assert original["gender"].isna().sum() == summary["B4"]


def test_natural_big_orders_exist(out):
    _, summary = out
    assert summary["C3"] == generate.ISSUE_RATES["C3_natural_big_orders"]


def test_formats_differ_between_branches(out):
    """ปัญหากลุ่ม A: แต่ละสาขาใช้รูปแบบต่างกันจริง"""
    root, _ = out
    nm = pd.read_csv(root / "data/raw/nimman_sales.csv", dtype=str)
    cm = pd.read_excel(root / "data/raw/cmu_sales.xlsx", dtype={"เลขที่บิล": str})
    tp = load_thaphae(root)
    assert nm["sale_datetime"].str.match(r"\d{2}/\d{2}/25\d{2} \d{2}:\d{2}").all()  # ปี พ.ศ.
    assert "ชื่อเมนู" in cm.columns and "ราคารวม" in cm.columns
    assert tp[0]["timestamp"].endswith("Z")  # UTC


def test_reproducible(out, tmp_path):
    """SEED คงที่ รันซ้ำต้องได้ไฟล์เหมือนเดิมทุกไบต์"""
    root1, _ = out
    generate.main(out_root=tmp_path, verbose=False)
    for name in ["nimman_sales.csv", "thaphae_sales.json", "members.csv"]:
        assert (root1 / "data/raw" / name).read_bytes() == (tmp_path / "data/raw" / name).read_bytes(), name


def test_member_names_unique_except_b3(out):
    """คนละคนต้องชื่อไม่ซ้ำ ชื่อที่ซ้ำ (หลังตัดช่องว่าง) ต้องมาจากการสมัครซ้ำ B3 เท่านั้น"""
    root, summary = out
    m = pd.read_csv(root / "data/raw/members.csv")
    assert m.iloc[:generate.N_MEMBERS]["full_name"].is_unique
    normalized = m["full_name"].str.split().str.join(" ")
    assert normalized.duplicated().sum() == summary["B3"]


def test_csv_has_bom_for_excel(out):
    """CSV ที่มีภาษาไทยต้องมี BOM เพื่อให้ Excel แสดงภาษาไทยถูกต้อง"""
    root, _ = out
    for name in ["members.csv", "products.csv", "holidays.csv", "nimman_sales.csv"]:
        assert (root / "data/raw" / name).read_bytes()[:3] == b"\xef\xbb\xbf", name
