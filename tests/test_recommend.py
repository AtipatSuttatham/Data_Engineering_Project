"""
ทดสอบโมดูล 7 (pipeline/recommend.py)
ใช้ฐานข้อมูลชั่วคราว (โหลดใหม่จาก pipeline) ไม่แตะไฟล์ doibrew.db จริง
"""

import numpy as np
import pandas as pd
import pytest

from evaluation import evaluate_recommend
from pipeline import load, recommend, transform


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("reco") / "test.db"
    load.run_load(transformed=transform.run_transform(save=False), db_path=path)
    return path


@pytest.fixture(scope="module")
def res(db):
    return recommend.run_recommend(db_path=db, save=True)


@pytest.fixture(scope="module")
def baskets(db):
    return recommend.load_baskets(db)


def test_formulas_match_manual_count(baskets):
    """Support / Confidence / Lift ต้องตรงกับการนับเองทีละบิล"""
    train, _ = recommend.split_by_time(baskets)
    pairs = recommend.mine_rules(train).set_index(["antecedent", "consequent"])
    a, b = "P02", "P12"   # อเมริกาโน่ -> ครัวซองต์
    n = len(train)
    has_a, has_b = train[a] == 1, train[b] == 1
    both = (has_a & has_b).sum()
    assert pairs.loc[(a, b), "support"] == pytest.approx(both / n)
    assert pairs.loc[(a, b), "confidence"] == pytest.approx(both / has_a.sum())
    assert pairs.loc[(a, b), "lift"] == pytest.approx((both / has_a.sum()) / (has_b.sum() / n))


def test_time_split_no_leakage(baskets):
    train, test = recommend.split_by_time(baskets)
    assert train["month"].max() < recommend.TEST_MONTH and (test["month"] == recommend.TEST_MONTH).all()
    assert len(train) + len(test) == len(baskets)


def test_rules_pass_thresholds(res):
    r = res["rules"]
    assert len(r) > 0
    assert (r["support"] >= recommend.MIN_SUPPORT).all()
    assert (r["confidence"] >= recommend.MIN_CONFIDENCE).all()
    assert (r["lift"] >= recommend.MIN_LIFT).all()


def test_recommend_excludes_items_in_bill(res):
    rec = recommend.recommend(["P02", "P12"], res["pairs"], res["popular"])
    assert not set(rec["product_id"]) & {"P02", "P12"}
    assert len(rec) == recommend.TOP_K and rec["product_id"].is_unique


def test_recommend_uses_rule_first(res):
    rec = recommend.recommend(["P02"], res["pairs"], res["popular"])
    assert rec.iloc[0]["product_id"] == "P12" and rec.iloc[0]["source"] == "rule"


def test_fallback_when_no_rule(res):
    """เมนูที่ไม่มีกฎ ต้องได้เมนูขายดีแทน และบอกว่าเป็นคำแนะนำสำรอง"""
    rec = recommend.recommend(["P09"], res["pairs"], res["popular"])   # ชามะนาว ไม่มีคู่
    assert (rec["source"] == "popular").all() and len(rec) == recommend.TOP_K


def test_model_beats_baseline(res):
    ev = res["evaluation"].set_index("method")
    model, base = ev.iloc[0], ev.iloc[1]
    assert model["hit_rate@1"] > base["hit_rate@1"]
    assert model[f"hit_rate@{recommend.TOP_K}"] > base[f"hit_rate@{recommend.TOP_K}"]


def test_evaluation_uses_only_multi_item_bills(baskets):
    _, test = recommend.split_by_time(baskets)
    cases = recommend.leave_one_out_cases(test)
    size = test[recommend.item_columns(test)].sum(axis=1)
    assert len(cases) == int(size[size >= 2].sum())
    assert all(hidden not in visible for visible, hidden in cases)


def test_rules_stable_across_time_and_branches(res):
    st = res["stability"]
    assert (abs(st["confidence_test"] - st["confidence"]) < 0.1).all()
    for c in [c for c in st.columns if c.startswith("lift_")]:
        assert (st[c] > 1).all()


def test_rules_saved_to_sqlite(res, db):
    saved = load.read_table("reco_rules", db)
    assert len(saved) == len(res["rules"])
    assert saved["antecedent_name"].notna().all()


def test_recommend_with_rules_read_from_sqlite(res, db):
    """หน้าเว็บอ่านกฎจาก SQLite (ไม่มีคอลัมน์ passes) ต้องแนะนำได้ผลเดียวกับกฎที่คำนวณสด"""
    from_db = load.read_table("reco_rules", db)
    for item in ["P01", "P02", "P03", "P07", "P09", "P12"]:
        a = recommend.recommend([item], from_db, res["popular"])
        b = recommend.recommend([item], res["pairs"], res["popular"])
        assert a["product_id"].tolist() == b["product_id"].tolist(), item


def test_finds_all_planted_pairs(res):
    ev = evaluate_recommend.evaluate(res).set_index("metric")["value"]
    assert ev["เจอถูก"] == ev["กฎที่ควรเจอ (คู่ที่ใส่ไว้ x 2 ทิศทาง)"]
    assert ev["กฎแปลกปลอม (ไม่ได้ใส่ไว้แต่ผ่านเกณฑ์)"] == 0


def test_reload_after_rules_exist(db):
    """โหลดข้อมูลใหม่ (โมดูล 6) หลังจากมีตาราง reco_rules แล้ว ต้องไม่ error เรื่อง Foreign key"""
    recommend.run_recommend(db_path=db, save=True)
    load.run_load(transformed=transform.run_transform(save=False), db_path=db)   # ต้องไม่ error
    tables = set(load.query("SELECT name FROM sqlite_master WHERE type = 'table'", db)["name"])
    assert "reco_rules" not in tables          # กฎเก่าถูกลบเพราะล้าสมัย
    recommend.run_recommend(db_path=db, save=True)
    assert len(load.read_table("reco_rules", db)) > 0
