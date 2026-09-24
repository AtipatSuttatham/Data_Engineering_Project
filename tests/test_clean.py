"""
ทดสอบโมดูล 4 (pipeline/clean.py) เทียบกับไฟล์เฉลยผ่าน evaluation/evaluate_cleaning.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import evaluate_cleaning
from pipeline import clean, extract, quality

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ext():
    return extract.run_extract(save=False)


@pytest.fixture(scope="module")
def res(ext):
    snapshot = {k: ext[k].copy() for k in ("sales", "members", "products")}
    r = clean.run_clean(save=False, ext=ext)
    r["_snapshot"] = snapshot
    return r


@pytest.fixture(scope="module")
def ev(res):
    return evaluate_cleaning.evaluate(res)


@pytest.fixture(scope="module")
def log():
    return json.loads((ROOT / "data/generator_log/injected_issues.json").read_text(encoding="utf-8"))


def metric(table, name):
    return table.set_index("metric").loc[name, "value"]


# ---- แถวที่ตัดออก ----
def test_removes_exactly_the_right_rows(ev):
    rows = ev["rows"]
    assert metric(rows, "ตัดถูก") == metric(rows, "แถวที่ควรตัด (B10 + C1)")
    assert metric(rows, "ตัดผิด (ไม่ควรตัดแต่ตัด)") == 0
    assert metric(rows, "แถวแปลกปลอม (ไม่มีในเฉลย)") == 0


def test_natural_big_orders_are_kept(res, log):
    """ออเดอร์ใหญ่จริง (C3) ต้องไม่ถูกตัด และต้องติดธง large_order_kept"""
    c3 = {r["receipt_id"] for r in log["issues"] if r["issue_id"] == "C3"}
    kept = res["sales"][res["sales"]["receipt_id"].isin(c3)]
    assert set(kept["receipt_id"]) == c3
    assert kept["clean_flags"].str.contains("large_order_kept").all()


def test_no_duplicates_left(res):
    s = res["sales"]
    assert not s.duplicated(subset=["receipt_id", "product_id"]).any()


# ---- ค่าที่แก้ / เติม ----
def test_all_fixes_are_correct(ev):
    fixes = ev["fixes"].dropna(subset=["accuracy"])
    assert (fixes["accuracy"] == 1.0).all(), fixes


def test_every_value_matches_truth(ev):
    assert (ev["mismatch"]["rows_not_matching_truth"] == 0).all(), ev["mismatch"]


def test_no_missing_values_needed_for_sales(res):
    s = res["sales"]
    for c in ["product_id", "qty", "unit_price", "line_total", "payment", "sale_datetime"]:
        assert s[c].notna().all(), c
    drinks = s["product_id"].map(res["products"].set_index("product_id")["category"]) != "bakery"
    assert s.loc[drinks, "size"].notna().all()
    assert (s["qty"] > 0).all()
    assert ((s["qty"] * s["unit_price"] - s["line_total"]).abs() < 1e-9).all()


def test_time_suspect_rows_kept_and_flagged(res, log):
    b11 = {r["receipt_id"] for r in log["issues"] if r["issue_id"] == "B11"}
    flagged = set(res["sales"].loc[res["sales"]["time_suspect"], "receipt_id"])
    assert flagged == b11


def test_member_ids_all_valid(res):
    mids = res["sales"]["member_id"].dropna()
    assert mids.isin(res["members"]["member_id"]).all()
    assert not mids.isin(res["member_id_map"]).any()  # ไม่มีรหัสของสมาชิกที่สมัครซ้ำเหลืออยู่


# ---- สมาชิก ----
def test_members_merged_and_standardized(res):
    m = res["members"]
    assert len(m) == 800
    assert m["member_id"].is_unique
    assert m["phone"].str.fullmatch(r"0\d{9}").all()
    assert (m["full_name"] == m["full_name"].str.split().str.join(" ")).all()


def test_birth_year_valid_and_flagged(res):
    m = res["members"]
    age = quality.DEFAULT_RECEIVED_AT.year - m["birth_year"]
    assert m["birth_year"].notna().all()
    assert age.between(quality.AGE_MIN, quality.AGE_MAX).all()
    assert m["birth_year_imputed"].dtype == bool and m["birth_year_imputed"].sum() > 0


def test_real_birth_years_untouched(ev):
    done, total = metric(ev["members"], "ปีเกิดที่ไม่ได้เติม ตรงเฉลย").split(" / ")
    assert done == total


def test_regression_beats_median_on_truth(ev):
    chosen = metric(ev["members"], "ปีเกิดที่เติม: คลาดเฉลี่ย (ปี) วิธีที่เลือก")
    median = metric(ev["members"], "ปีเกิดที่เติม: คลาดเฉลี่ย (ปี) ถ้าใช้ Median")
    assert chosen < median


def test_gender_unknown_not_guessed(res):
    m = res["members"]
    assert set(m["gender"]) <= set(quality.GENDER_DOMAIN)
    assert (m["gender"] == quality.GENDER_UNKNOWN).sum() > 0


def test_imputation_table_picks_lowest_error(res):
    t = res["imputation"].dropna(subset=["mae_years"])
    assert len(t) == 4
    assert t.loc[t["chosen"], "mae_years"].iloc[0] == t["mae_years"].min()


# ---- outlier ----
def test_outlier_methods_compared(ev):
    o = ev["outliers"].set_index("method")
    assert o.loc["final (Z-score + > 50 แก้ว)", "C1_typos_caught"] == o.loc["zscore", "C1_total"]
    assert o.loc["final (Z-score + > 50 แก้ว)", "C3_real_orders_flagged"] == 0
    assert o.loc["iqr", "normal_rows_flagged"] > 1000  # IQR ใช้ไม่ได้กับข้อมูลที่ค่าส่วนใหญ่เป็น 1


# ---- ตรวจซ้ำ (Validation) ----
def test_quality_after_cleaning(res):
    after = res["quality_after"].rules.set_index("rule_id")["score"]
    for rule in ["CP2", "CP3", "CP4", "CP6", "UQ1", "UQ2", "UQ3", "UQ4", "VL1", "VL3", "VL4", "VL5",
                 "CS1", "CS2", "CS3", "AC1"]:
        assert after[rule] == 1.0, rule
    assert res["quality_after"].overall["score_avg"] > res["quality_before"].overall["score_avg"]


def test_cleaning_does_not_modify_input(ext, res):
    """run_clean ต้องไม่แก้ตารางที่ส่งเข้ามา (เก็บต้นฉบับไว้แสดง Before / After ได้)"""
    for k in ("sales", "members", "products"):
        pd.testing.assert_frame_equal(ext[k], res["_snapshot"][k])


def test_pipeline_does_not_read_answer_key():
    """กฎใน CLAUDE.md: โค้ดใน pipeline/ ห้ามอ่านไฟล์เฉลย"""
    for f in (ROOT / "pipeline").glob("*.py"):
        assert "generator_log" not in f.read_text(encoding="utf-8"), f.name


# ---- จุดที่อาจพังเมื่อข้อมูลเปลี่ยน ----
def _cleaning_log():
    return clean.CleaningLog()


def test_size_from_price_tolerates_float_division(ext):
    """ราคาที่ได้จากการหาร (เช่น 64.9999999) ยังต้องย้อนหาขนาดได้"""
    s = ext["sales"].copy()
    s["clean_flags"] = ""
    row = s[(s["product_id"] == "P03") & (s["size"] == "M") & s["unit_price"].notna()].index[0]
    s.loc[row, "size"] = None
    s.loc[row, "unit_price"] = 65 - 1e-9
    out = clean.fill_missing_sales(s.loc[[row]], ext["products"], _cleaning_log(), [])
    assert out.loc[row, "size"] == "M"


def test_missing_size_and_qty_together(ext):
    """ถ้าขาดทั้งขนาดและจำนวน ต้องเติมขนาดด้วย Mode ก่อน แล้วจึงคำนวณจำนวนได้"""
    s = ext["sales"].copy()
    s["clean_flags"] = ""
    base = s[(s["branch"] == "cmu") & s["qty"].notna() & (s["size"] == "M") & (s["product_id"] == "P03")]
    row = base.index[0]
    s.loc[row, ["size", "qty", "unit_price"]] = [None, np.nan, np.nan]
    s.loc[row, "line_total"] = 65.0  # ลาเต้ M 1 แก้ว
    out = clean.fill_missing_sales(s, ext["products"], _cleaning_log(), [])
    assert out.loc[row, "qty"] == 1 and out.loc[row, "size"] == "M"
    assert "size_mode" in out.loc[row, "clean_flags"]


def test_over_limit_without_zscore_is_flagged_for_review(ext):
    """เกินกฎ 50 แก้วแต่ Z-score ไม่ว่าแปลก ต้องไม่ถูกปล่อยผ่านเงียบ ๆ"""
    s = ext["sales"].iloc[:3].copy()
    s["clean_flags"] = ""
    flags = pd.DataFrame({"qty": [60.0, 1.0, 1.0], "zscore": [False, False, False]}, index=s.index)
    out = clean.remove_qty_typos(s, flags, _cleaning_log(), [])
    assert "qty_over_limit_review" in out.iloc[0]["clean_flags"]
