"""
ทดสอบโมดูล 8 (pipeline/forecast.py) เน้นเรื่องการกัน Data leakage และความถูกต้องของตัวแปร
ใช้ฐานข้อมูลชั่วคราว ไม่แตะไฟล์ doibrew.db จริง
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import forecast as fc
from pipeline import load, recommend, transform


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("fc") / "test.db"
    load.run_load(transformed=transform.run_transform(save=False), db_path=path)
    return path


@pytest.fixture(scope="module")
def df(db):
    return fc.load_features(db)


@pytest.fixture(scope="module")
def res(db):
    return fc.run_forecast(db_path=db, save=True)


# ---- กัน Data leakage ----
def test_scaler_uses_training_data_only(df):
    train, test = df[df["date"].dt.month < 6], df[df["date"].dt.month == 6]
    scaler = fc.fit_scaler(train)
    tr, te = fc.apply_scaler(train, scaler), fc.apply_scaler(test, scaler)
    assert abs(tr["lag7_revenue_z"].mean()) < 1e-9          # ข้อมูลฝึก: ค่าเฉลี่ย 0 พอดี
    assert abs(te["trend_z"].mean()) > 0.5                   # ข้อมูลทดสอบ: ไม่ได้ปรับให้เฉลี่ย 0 (ไม่ได้ใช้ค่าของมันเอง)


def test_model_selection_ignores_june(df):
    """ถ้าเปลี่ยนยอดขายเดือน มิ.ย. การเลือกโมเดลต้องได้ผลเดิม (เลือกจาก พ.ค. เท่านั้น)"""
    chosen, val = fc.select_model(df)
    changed = df.copy()
    june = changed["date"].dt.month == 6
    changed.loc[june, "revenue"] = np.random.default_rng(0).uniform(0, 10000, june.sum())
    chosen2, val2 = fc.select_model(changed)
    assert chosen == chosen2
    pd.testing.assert_frame_equal(val, val2)


def test_predictions_do_not_use_future_rows(df):
    """ผลทำนายเดือน เม.ย. ต้องไม่เปลี่ยน ถ้าข้อมูล พ.ค.-มิ.ย. เปลี่ยน"""
    base = fc.predict_all(df[df["date"].dt.month < 4], df[df["date"].dt.month == 4])
    changed = df.copy()
    changed.loc[changed["date"].dt.month >= 5, "revenue"] *= 3
    again = fc.predict_all(changed[changed["date"].dt.month < 4], changed[changed["date"].dt.month == 4])
    pd.testing.assert_frame_equal(base, again)


def test_no_missing_days_in_training(df):
    assert df["revenue"].notna().all() and df["lag7_revenue"].notna().all()


# ---- ตัวแปรและโมเดล ----
@pytest.mark.parametrize("name", list(fc.FEATURES))
def test_no_dummy_variable_trap(df, name):
    """ตัวแปรทุกตัวต้องไม่ซ้ำซ้อนกัน (เมทริกซ์ full rank) และโมเดลไม่มีค่าคงที่"""
    X = fc.apply_scaler(df, fc.fit_scaler(df))[fc.FEATURES[name]].to_numpy(dtype=float)
    assert np.linalg.matrix_rank(X) == X.shape[1]
    model = fc.fit_model(fc.apply_scaler(df, fc.fit_scaler(df)), name)
    assert model.intercept_ == 0


def test_interactions(df):
    row = df[(df["branch"] == "cmu") & (df["is_holiday"] == 1)].iloc[0]
    assert row["holiday_cmu"] == 1 and row["holiday_nimman"] == 0 and row["holiday_thaphae"] == 0
    hol = df[[f"holiday_{b}" for b in fc.BRANCHES]].sum(axis=1)
    assert (hol == df["is_holiday"]).all()


def test_walk_forward_is_nested(df):
    """ผลของเดือน เม.ย. ใน Walk-forward ต้องไม่เปลี่ยน ถ้าข้อมูล พ.ค.-มิ.ย. เปลี่ยน (ไม่ใช้อนาคตเลือกโมเดล)"""
    base = fc.walk_forward(df).set_index("test_month").loc[4]
    changed = df.copy()
    changed.loc[changed["date"].dt.month >= 5, "revenue"] *= 3
    again = fc.walk_forward(changed).set_index("test_month").loc[4]
    assert base["chosen_model"] == again["chosen_model"] and base["MAE_model"] == again["MAE_model"]


def test_metrics_reject_missing_predictions():
    with pytest.raises(ValueError):
        fc.metrics(pd.Series([1.0, 2.0]), pd.Series([1.0, np.nan]))


def test_metrics_formula():
    m = fc.metrics(pd.Series([100.0, 200.0, 300.0]), pd.Series([110.0, 190.0, 330.0]))
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["bias"] == pytest.approx(10.0)


# ---- ผลลัพธ์ ----
def test_results_structure(res, df):
    assert res["chosen"] in fc.FEATURES
    assert set(res["test"]["method"]) == set(fc.METHOD_TH)
    assert list(res["walk_forward"]["test_month"]) == [4, 5, 6]
    assert res["n_train"] + res["n_test"] == len(df)            # ฝึก (ม.ค.-พ.ค.) + ทดสอบ (มิ.ย.) = ข้อมูลทั้งหมด
    assert res["n_test"] == (df["date"].dt.month == 6).sum()


def test_next_week_forecast(res, db):
    nw = res["next_week"]
    assert len(nw) == fc.FORECAST_DAYS * len(fc.BRANCHES)
    assert nw["date"].min() == pd.Timestamp("2026-07-01") and nw["date"].max() == pd.Timestamp("2026-07-07")
    assert nw["predicted_revenue"].notna().all()
    # ยอด 7 วันก่อนต้องมาจาก agg_daily จริง
    daily = load.query("SELECT branch, date, revenue FROM agg_daily WHERE date = '2026-06-24'", db).set_index("branch")
    first = nw[nw["date"] == pd.Timestamp("2026-07-01")].set_index("branch")
    for b in fc.BRANCHES:
        assert first.loc[b, "lag7_revenue"] == daily.loc[b, "revenue"]


def test_tables_saved_and_reload_still_works(res, db):
    """ผลลัพธ์อยู่ใน SQLite และรันโมดูล 6 ซ้ำได้ (ตารางอยู่ใน DERIVED_TABLES)"""
    for t in ["forecast_daily", "forecast_coefficients", "forecast_next_week"]:
        assert len(load.read_table(t, db)) > 0
        assert t in load.DERIVED_TABLES
    recommend.run_recommend(db_path=db, save=True)
    load.run_load(transformed=transform.run_transform(save=False), db_path=db)   # ต้องไม่ error
    fc.run_forecast(db_path=db, save=True)
