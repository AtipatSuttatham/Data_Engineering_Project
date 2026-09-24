"""
ทดสอบโมดูล 5 (pipeline/transform.py)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import clean, transform

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cleaned():
    return clean.run_clean(save=False)


@pytest.fixture(scope="module")
def res(cleaned):
    return transform.run_transform(save=False, cleaned=cleaned)


# ---- ยอดขายต้องไม่หายระหว่าง transform ----
def test_revenue_preserved_in_every_table(cleaned, res):
    total = cleaned["sales"]["line_total"].sum()
    assert res["fact_sales"]["line_total"].sum() == pytest.approx(total)
    assert res["agg_daily"]["revenue"].sum() == pytest.approx(total)
    assert res["agg_product"]["revenue"].sum() == pytest.approx(total)
    assert res["dim_member"]["total_spent"].sum() == pytest.approx(
        cleaned["sales"].loc[cleaned["sales"]["member_id"].notna(), "line_total"].sum())


def test_hourly_excludes_time_suspect(cleaned, res):
    s = cleaned["sales"]
    assert res["agg_hourly"]["revenue"].sum() == pytest.approx(s.loc[~s["time_suspect"], "line_total"].sum())


# ---- T4 Anonymization ----
def test_no_personal_data_in_any_table(cleaned, res):
    names = set(cleaned["members"]["full_name"])
    phones = set(cleaned["members"]["phone"])
    for name in transform.TABLES:
        df = res[name]
        assert "full_name" not in df.columns and "phone" not in df.columns, name
        values = set(df.astype(str).to_numpy().ravel())
        assert not (values & names), name
        assert not (values & phones), name


def test_phone_hash_is_consistent(cleaned, res):
    h = transform.hash_phone(cleaned["members"]["phone"])
    assert (h.to_numpy() == res["dim_member"]["phone_hash"].to_numpy()).all()
    assert res["dim_member"]["phone_hash"].is_unique
    assert res["dim_member"]["phone_hash"].str.fullmatch(r"[0-9a-f]{64}").all()


# ---- T8 Star schema: กุญแจเชื่อมตารางครบ ----
def test_foreign_keys_resolve(res):
    f = res["fact_sales"]
    assert f["date_key"].isin(res["dim_date"]["date_key"]).all()
    assert f["product_id"].isin(res["dim_product"]["product_id"]).all()
    assert f["branch_id"].isin(res["dim_branch"]["branch_id"]).all()
    assert f["member_id"].dropna().isin(res["dim_member"]["member_id"]).all()
    assert f["sale_line_id"].is_unique


# ---- T2, T3 ----
def test_discretization(res):
    m = res["dim_member"]
    assert set(m["age_group"]) <= set(transform.AGE_LABELS)
    assert (m["age"] <= 22).eq(m["age_group"] == "<=22").all()
    buyers = m[m["spend_tier"] != "ไม่เคยซื้อ"]
    counts = buyers["spend_tier"].value_counts()
    assert counts.max() - counts.min() <= 1  # Equal-frequency: แต่ละกลุ่มมีคนเท่ากัน
    # กลุ่มสูงต้องใช้จ่ายมากกว่ากลุ่มต่ำ
    assert buyers.groupby("spend_tier")["total_spent"].min()["สูง"] >= buyers.groupby("spend_tier")["total_spent"].max()["ต่ำ"]


def test_time_of_day(res):
    f = res["fact_sales"]
    assert f.loc[f["time_suspect"], "time_of_day"].isna().all()
    ok = f[~f["time_suspect"]]
    assert ok["time_of_day"].notna().all()
    assert (ok.loc[ok["hour"].between(7, 10), "time_of_day"] == "เช้า").all()


def test_member_rollup(cleaned, res):
    m = res["dim_member"].set_index("member_id")
    s = cleaned["sales"]
    one = s["member_id"].dropna().iloc[0]
    assert m.loc[one, "n_bills"] == s.loc[s["member_id"] == one, "receipt_id"].nunique()


# ---- T5 วันที่ไม่มีข้อมูล ----
def test_missing_days_vs_zero_days(res):
    """ท่าแพ 3 วัน (POS ออฟไลน์) = ค่าว่าง / มช. ช่วงปิดเทอมที่ไม่มีลูกค้า = 0 (ตามการวิเคราะห์ในโมดูล 3)"""
    log = json.loads((ROOT / "data/generator_log/injected_issues.json").read_text(encoding="utf-8"))
    offline = {r["date"] for r in log["issues"] if r["issue_id"] == "B15"}
    d = res["agg_daily"]
    missing = d[d["data_missing"]]
    assert set(missing["branch"]) == {"thaphae"}
    assert {str(x) for x in missing["date"]} == offline
    assert missing["revenue"].isna().all()
    zero = d[(d["bills"] == 0)]
    assert set(zero["branch"]) == {"cmu"} and (zero["revenue"] == 0).all()


def test_missing_rule_robust_to_window(cleaned):
    """กฎแยกข้อมูลหายต้องให้ผลเดิมแม้เปลี่ยนขนาดช่วงเวลา"""
    s = transform.add_sale_attributes(cleaned["sales"], cleaned["holidays"])
    dd = transform.build_dim_date(cleaned["holidays"])
    original = transform.MISSING_WINDOW_DAYS
    results = set()
    try:
        for w in (7, 14, 21, 28):
            transform.MISSING_WINDOW_DAYS = w
            d = transform.build_agg_daily(s, dd)
            results.add(tuple(sorted((r.branch, str(r.date)) for r in d[d["data_missing"]].itertuples())))
    finally:
        transform.MISSING_WINDOW_DAYS = original
    assert len(results) == 1


def test_basket(cleaned, res):
    b = res["basket"]
    items = b.drop(columns=["receipt_id", "branch"])
    assert set(np.unique(items.to_numpy())) <= {0, 1}
    assert len(b) == cleaned["sales"]["receipt_id"].nunique()
    assert items.to_numpy().sum() == len(cleaned["sales"])  # 1 บิลไม่มีเมนูซ้ำ


# ---- T6 Encoding + T7 Normalization ----
def test_encoding(res):
    mf = res["model_features"]
    onehot = [c for c in mf.columns if c.startswith("branch_")]
    dummy = [c for c in mf.columns if c.startswith("dow_")]
    assert len(onehot) == 3 and (mf[onehot].sum(axis=1) == 1).all()           # One-hot: 1 คอลัมน์เป็น 1 เสมอ
    assert len(dummy) == 6 and "dow_Mon" not in dummy                          # Dummy: N-1 คอลัมน์
    assert (mf[dummy].sum(axis=1) <= 1).all()
    f = res["fact_sales"]
    assert f.loc[f["size"] == "S", "size_ordinal"].eq(1).all() and f.loc[f["size"] == "L", "size_ordinal"].eq(3).all()


def test_normalization(res):
    d = res["agg_daily"]
    mm = d.groupby("branch")["revenue_minmax"]
    assert np.allclose(mm.min(), 0) and np.allclose(mm.max(), 1)
    mf = res["model_features"]
    for c in ["lag7_revenue_z", "trend_z"]:
        assert abs(mf[c].mean()) < 1e-9 and abs(mf[c].std(ddof=0) - 1) < 1e-9


def test_model_features_exclude_missing_days(res):
    mf = res["model_features"]
    assert mf["revenue"].notna().all() and mf["lag7_revenue"].notna().all()
    missing = res["agg_daily"].loc[res["agg_daily"]["data_missing"], ["branch", "date"]]
    keys = set(zip(mf["branch"], mf["date"]))
    assert not any((b, dt) in keys for b, dt in zip(missing["branch"], missing["date"]))


def test_transform_does_not_modify_input(cleaned):
    before = cleaned["sales"].copy()
    transform.run_transform(save=False, cleaned=cleaned)
    pd.testing.assert_frame_equal(before, cleaned["sales"])


# ---- จุดที่อาจพังเมื่อข้อมูลเปลี่ยน ----
def test_missing_age_gives_missing_group(cleaned):
    m = cleaned["members"].copy()
    m.loc[m.index[0], "birth_year"] = pd.NA
    s = transform.add_sale_attributes(cleaned["sales"], cleaned["holidays"])
    dm = transform.build_dim_member(m, s, cleaned["products"])
    assert pd.isna(dm.loc[0, "age_group"]) and pd.isna(dm.loc[0, "age_group_ordinal"])
    assert "nan" not in set(dm["age_group"].dropna())


def test_min_max_constant_series():
    out = transform.min_max(pd.Series([5.0, 5.0, 5.0]))
    assert (out == 0).all()


def test_agg_product_covers_all_months(cleaned):
    s = transform.add_sale_attributes(cleaned["sales"], cleaned["holidays"])
    no_june = s[s["month"] != 6]
    out = transform.build_agg_product(no_june, cleaned["products"])
    assert set(out["month"]) == set(range(1, 7))
    assert (out.loc[out["month"] == 6, "qty"] == 0).all()


def test_spend_tier_with_few_buyers(cleaned):
    s = transform.add_sale_attributes(cleaned["sales"], cleaned["holidays"])
    one_buyer = s[s["member_id"] == s["member_id"].dropna().iloc[0]]
    dm = transform.build_dim_member(cleaned["members"], one_buyer, cleaned["products"])
    assert set(dm["spend_tier"]) == {"ไม่เคยซื้อ"}
