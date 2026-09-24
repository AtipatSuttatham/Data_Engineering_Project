"""
ทดสอบโมดูล 3 (pipeline/quality.py)
ตรวจว่ากฎคุณภาพจับปัญหาที่ใส่ไว้ได้ครบ (Recall) และไม่จับผิดตัว (Precision) โดยเทียบกับไฟล์เฉลย
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline import extract, quality

ROOT = Path(__file__).resolve().parents[1]
N_MEMBERS = 800  # สมาชิกตัวจริง แถวหลังจากนี้คือสมาชิกที่สมัครซ้ำ (B3)


@pytest.fixture(scope="module")
def run():
    ext = extract.run_extract(save=False)
    before = {k: ext[k].copy() for k in ("sales", "members", "products")}
    res = quality.assess(ext["sales"], ext["members"], ext["products"],
                         received_at=quality.received_at_from_meta(ext["thaphae_meta"]),
                         extract_report=ext["report"])
    return res, ext, before


@pytest.fixture(scope="module")
def log():
    return json.loads((ROOT / "data/generator_log/injected_issues.json").read_text(encoding="utf-8"))


def flagged(res, rule_id):
    return res.details[res.details["rule_id"] == rule_id]


def not_duplicate_copy(res, rows):
    """ตัดแถวที่เป็นสำเนาจากข้อมูลซ้ำ (UQ1) ออก เพื่อนับเฉพาะตัวต้นฉบับ"""
    dup_rows = set(flagged(res, "UQ1")["row"].astype(int))
    return rows[~rows["row"].astype(int).isin(dup_rows)]


def injected(log, issue_id, field="receipt_id"):
    return [r[field] for r in log["issues"] if r["issue_id"] == issue_id]


@pytest.mark.parametrize("rule_id, issue_id", [
    ("CP3", "B8"), ("CP4", "B7"), ("VL1", "B10"), ("CS1", "B14"), ("AC1", "C2"),
])
def test_sales_rules_match_injected_count(run, log, rule_id, issue_id):
    """จำนวนแถวที่ผิด (ไม่นับสำเนาจากข้อมูลซ้ำ) ต้องเท่ากับจำนวนที่ใส่ไว้ และอยู่ในบิลเดียวกับที่ใส่ไว้"""
    res = run[0]
    rows = not_duplicate_copy(res, flagged(res, rule_id))
    assert len(rows) == log["summary"][issue_id]
    assert set(rows["key"]) == set(injected(log, issue_id))


def test_missing_price_covers_b6_and_b8(run, log):
    res = run[0]
    keys = set(flagged(res, "CP2")["key"])
    assert set(injected(log, "B6")) <= keys and set(injected(log, "B8")) <= keys


@pytest.mark.parametrize("rule_id, issue_id", [("CP5", "B4"), ("CP6", "B5"), ("VL4", "B9")])
def test_member_rules_match_injected(run, log, rule_id, issue_id):
    """นับเฉพาะสมาชิกตัวจริง (แถวที่สมัครซ้ำคัดลอกค่ามาจากแถวเดิมด้วย)"""
    rows = flagged(run[0], rule_id)
    original = rows[rows["row"].astype(int) < N_MEMBERS]
    assert set(original["key"]) == set(injected(log, issue_id, "member_id"))


def test_duplicate_member_rule(run, log):
    assert set(flagged(run[0], "UQ4")["key"]) == set(injected(log, "B3", "member_id"))


def test_duplicate_rows_and_bills(run, log):
    res = run[0]
    uq1 = flagged(res, "UQ1")
    assert (uq1["branch"] == "nimman").sum() == log["summary"]["B1"]
    assert set(uq1[uq1["branch"] == "thaphae"]["key"]) == set(injected(log, "B2"))
    assert set(flagged(res, "UQ2")["key"]) == set(injected(log, "B2"))


def test_bill_level_rules(run, log):
    res = run[0]
    assert set(flagged(res, "VL2")["key"]) == set(injected(log, "B11"))
    assert set(flagged(res, "CS2")["key"]) == set(injected(log, "B12"))


def test_population_completeness_finds_offline_days(run, log):
    cp7 = flagged(run[0], "CP7")
    thaphae_days = {k.split()[1] for k in cp7[cp7["branch"] == "thaphae"]["key"]}
    assert thaphae_days == set(injected(log, "B15", "date"))


def test_optional_fields_not_counted_as_missing(run):
    """บิลไม่มีสมาชิก (ลูกค้าทั่วไป) และเบเกอรี่ไม่มีขนาด ต้องไม่ถูกนับว่าข้อมูลหาย"""
    res, ext, _ = run
    s = ext["sales"]
    drinks = s["product_id"].map(ext["products"].set_index("product_id")["category"]).isin(["coffee", "non_coffee"])
    rules = res.rules.set_index("rule_id")
    assert rules.loc["CP4", "checked"] == drinks.sum()
    assert rules.loc["CS2", "checked"] == s["member_id"].notna().sum()


def test_score_formulas(run):
    res = run[0]
    r = res.rules.dropna(subset=["violations"])
    assert ((1 - r["violations"] / r["checked"]) - r["score"]).abs().max() < 1e-12
    assert res.rules["score"].between(0, 1).all()
    d = res.dimensions.set_index("dimension")
    for dim, g in res.rules.groupby("dimension"):
        assert abs(d.loc[dim, "score_avg"] - g["score"].mean()) < 1e-12
        assert d.loc[dim, "score_min"] == g["score"].min()
    assert abs(res.overall["score_avg"] - res.dimensions["score_avg"].mean()) < 1e-12


def test_timeliness_formula(run):
    """ข้อมูลล่าสุด 30 มิ.ย. รับข้อมูล 1 ก.ค. -> Currency 1 วัน -> 1 - 1/7"""
    tm1 = run[0].rules.set_index("rule_id").loc["TM1", "score"]
    assert abs(tm1 - (1 - 1 / quality.VOLATILITY_DAYS)) < 1e-12


def test_assess_does_not_modify_data(run):
    """โมดูล 3 วัดอย่างเดียว ห้ามแก้ข้อมูล"""
    _, ext, before = run
    for k, df in before.items():
        pd.testing.assert_frame_equal(ext[k], df)


def test_compare_same_data_has_no_change(run):
    res = run[0]
    assert (quality.compare(res, res)["score_change"] == 0).all()


def _assess(ext, sales=None, members=None, received_at=None):
    return quality.assess(ext["sales"] if sales is None else sales,
                          ext["members"] if members is None else members,
                          ext["products"], received_at=received_at)


def test_missing_column_is_reported_without_crash(run):
    """ถ้าคอลัมน์หาย กฎ CP1 ต้องรายงานได้ (ไม่ error)"""
    ext = run[1]
    res = _assess(ext, sales=ext["sales"].drop(columns=["datetime_raw"]))
    cp1 = flagged(res, "CP1")
    assert list(cp1["key"]) == ["datetime_raw"]


def test_whole_branch_missing_is_detected(run):
    """ถ้าข้อมูลหายทั้งสาขา กฎ CP7 ต้องจับได้ครบ 181 วัน"""
    ext = run[1]
    no_cmu = ext["sales"][ext["sales"]["branch"] != "cmu"]
    cp7 = flagged(_assess(ext, sales=no_cmu), "CP7")
    assert (cp7["branch"] == "cmu").sum() == 181


def test_timeliness_never_above_one(run):
    """ถ้าวันที่รับข้อมูลเก่ากว่าข้อมูลล่าสุด คะแนนต้องไม่เกิน 100%"""
    res = _assess(run[1], received_at=pd.Timestamp("2026-06-01", tz="Asia/Bangkok"))
    assert res.rules.set_index("rule_id").loc["TM1", "score"] <= 1


def test_price_tolerance_for_division(run):
    """ราคาที่มีทศนิยมเล็กน้อยจากการหาร (เช่น 65.0000001) ต้องถือว่าตรงตารางราคา"""
    ext = run[1]
    s = ext["sales"].copy()
    s["unit_price"] = s["unit_price"] + 1e-7
    assert flagged(_assess(ext, sales=s), "AC1").shape[0] == flagged(run[0], "AC1").shape[0]
