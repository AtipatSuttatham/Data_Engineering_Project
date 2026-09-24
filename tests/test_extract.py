"""
ทดสอบโมดูล 2 (pipeline/extract.py)
ใช้ไฟล์เฉลยใน data/generator_log/ ได้ เพราะเป็น tests (กฎใน CLAUDE.md หัวข้อ 7)
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline import extract

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "generator_log"


@pytest.fixture(scope="module")
def res():
    return extract.run_extract(save=False)


@pytest.fixture(scope="module")
def issues():
    return json.loads((LOG_DIR / "injected_issues.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truth():
    t = pd.read_csv(LOG_DIR / "sales_truth.csv", encoding="utf-8-sig", parse_dates=["sale_dt"])
    t["sale_dt"] = t["sale_dt"].dt.tz_localize("Asia/Bangkok")
    return t


def test_reconciliation(res):
    """ตรวจยอด: แถวที่ออก = แถวที่อ่านเข้า (ไม่หาย ไม่เกิน)"""
    rep = res["report"]
    assert rep["reconciled"].all()
    raw = ROOT / "data" / "raw"
    n_nimman = len(pd.read_csv(raw / "nimman_sales.csv"))
    n_cmu = len(pd.read_excel(raw / "cmu_sales.xlsx"))
    receipts = json.loads((raw / "thaphae_sales.json").read_text(encoding="utf-8"))["receipts"]
    n_thaphae = sum(len(r["items"]) for r in receipts)
    assert len(res["sales"]) == n_nimman + n_cmu + n_thaphae


def test_schema(res):
    assert list(res["sales"].columns) == extract.STAGING_COLUMNS


def test_standard_vocab(res):
    s = res["sales"]
    assert set(s["size"].dropna()) <= {"S", "M", "L"}
    assert set(s["payment"].dropna()) <= {"cash", "qr", "card"}
    assert s["payment"].notna().all()
    assert set(s["branch"]) == {"nimman", "cmu", "thaphae"}


def test_dates_are_2026_thai_time(res):
    """ปี พ.ศ. ต้องแปลงเป็น 2026 และทุกค่าต้องเป็นเวลาไทย"""
    dt = res["sales"]["sale_datetime"]
    assert dt.notna().all()
    assert (dt.dt.year == 2026).all()
    assert str(dt.dt.tz) == "Asia/Bangkok"


def test_datetime_matches_truth(res, truth, issues):
    """เวลาทุกบิลต้องตรงกับเฉลย (ยกเว้นบิลที่ใส่เวลาผิดไว้ B11) ตรวจว่าแปลง UTC และ พ.ศ. ถูก"""
    b11 = {r["receipt_id"] for r in issues["issues"] if r["issue_id"] == "B11"}
    got = res["sales"].groupby("receipt_id")["sale_datetime"].first()
    exp = truth.groupby("receipt_id")["sale_dt"].first()
    common = exp.index.difference(list(b11))
    assert (got.loc[common] == exp.loc[common]).all()


def test_product_unmatched_only_b14(res, issues):
    """จับคู่สินค้าไม่ได้เฉพาะแถวที่ชื่อเมนูสะกดผิด (B14) เท่านั้น"""
    s = res["sales"]
    b14 = {r["injected"] for r in issues["issues"] if r["issue_id"] == "B14"}
    unmatched = s[s["product_id"].isna()]
    assert len(unmatched) == issues["summary"]["B14"]
    assert set(unmatched["product_raw"]) <= b14
    assert unmatched["extract_flags"].str.contains("product_unmatched").all()


def test_products_match_truth(res, truth):
    """รายการสินค้าในแต่ละบิล (ตัดแถวซ้ำและแถวสะกดผิด) ต้องเป็นสินค้าจากเฉลย"""
    s = res["sales"].dropna(subset=["product_id"])
    got = s.groupby("receipt_id")["product_id"].apply(set)
    exp = truth.groupby("receipt_id")["product_id"].apply(set)
    assert all(got[r] <= exp[r] for r in got.index)


def test_cmu_unit_price_derived(res):
    """มช. ต้องคำนวณราคาต่อชิ้นจากราคารวม ÷ จำนวน"""
    cmu = res["sales"][res["sales"]["branch"] == "cmu"]
    ok = cmu.dropna(subset=["unit_price"])
    assert (ok["unit_price"] * ok["qty"] == ok["line_total"]).all()
    assert (ok["price_source"] == "derived_from_total").all()


def test_does_not_remove_duplicates(res, issues):
    """โมดูล 2 ต้องไม่ลบข้อมูลซ้ำเอง (ให้โมดูล 3 วัด และโมดูล 4 ลบ)"""
    s = res["sales"]
    core = [c for c in extract.STAGING_COLUMNS if c != "source_row"]
    nm = s[s["branch"] == "nimman"]
    assert nm.duplicated(subset=core).sum() >= issues["summary"]["B1"]
    tp_bills = s[s["branch"] == "thaphae"]["source_row"].str.extract(r"receipts\[(\d+)\]")[0].nunique()
    assert tp_bills == res["thaphae_meta"]["receipts"]


def test_members_phone_keeps_leading_zero(res):
    phones = res["members"]["phone"]
    assert phones.map(type).eq(str).all()
    local = phones[~phones.str.startswith("+")]
    assert local.str.startswith("0").all()


def test_members_birth_year_is_integer(res):
    assert str(res["members"]["birth_year"].dtype) == "Int64"


def test_report_total_row(res):
    """แถว total ต้องรวมเฉพาะคอลัมน์ที่นับหน่วยเดียวกัน (ระเบียนในไฟล์นับคนละหน่วย จึงเว้นว่าง)"""
    total = res["report"].set_index("branch").loc["total"]
    assert pd.isna(total["records_in_file"])
    assert total["rows_read"] == len(res["sales"])


def test_missing_values_use_nan_not_none(res):
    """ค่าว่างจาก JSON (None) ต้องเป็นแบบเดียวกับไฟล์อื่น (NaN)"""
    tp = res["sales"][res["sales"]["branch"] == "thaphae"]
    for c in ["size_raw", "member_id"]:
        assert not tp[c].map(lambda v: v is None).any(), c
