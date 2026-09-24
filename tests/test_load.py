"""
ทดสอบโมดูล 6 (pipeline/load.py) ใช้ฐานข้อมูลชั่วคราว ไม่แตะไฟล์ doibrew.db จริง
"""

import sqlite3

import pandas as pd
import pytest

from pipeline import load, transform


@pytest.fixture(scope="module")
def tables():
    return transform.run_transform(save=False)


@pytest.fixture(scope="module")
def loaded(tables, tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    res = load.run_load(transformed=tables, db_path=db)
    return res, db


@pytest.fixture
def con(loaded):
    """เชื่อมต่อฐานข้อมูลทดสอบ แล้วยกเลิกการเปลี่ยนแปลงและปิดทุกครั้งหลังจบ test (กันฐานข้อมูลถูกล็อก)"""
    c = sqlite3.connect(loaded[1])
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.rollback()
    c.close()


# ---- Load verification ----
def test_verification_all_passed(loaded):
    v = loaded[0]["verification"]
    assert v["passed"].all(), v[~v["passed"]]


def test_row_counts_and_revenue(con, tables):
    for t in load.LOAD_ORDER:
        assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == len(tables[t]), t
    total = con.execute("SELECT SUM(line_total) FROM fact_sales").fetchone()[0]
    assert total == pytest.approx(tables["fact_sales"]["line_total"].sum())


def test_datetime_round_trip(loaded, tables):
    back = load.read_table("fact_sales", loaded[1]).sort_values("sale_line_id")
    assert back["sale_datetime"].str.endswith("+07:00").all()
    parsed = pd.to_datetime(back["sale_datetime"], format="ISO8601").reset_index(drop=True)
    src = tables["fact_sales"].sort_values("sale_line_id")["sale_datetime"].reset_index(drop=True)
    assert (parsed == src).all()


def test_missing_days_stay_null(loaded):
    """วันที่ข้อมูลหาย (ท่าแพ 3 วัน) ต้องเป็น NULL ในฐานข้อมูล ไม่ใช่ 0"""
    df = load.query("SELECT branch, revenue FROM agg_daily WHERE data_missing = 1", loaded[1])
    assert len(df) == 3 and df["revenue"].isna().all() and set(df["branch"]) == {"thaphae"}


def test_view_joins_all_rows(loaded, tables):
    n = load.query("SELECT COUNT(*) n FROM v_sales_enriched", loaded[1])["n"].iloc[0]
    assert n == len(tables["fact_sales"])


def test_no_personal_data(con):
    cols = {r[1] for t in load.LOAD_ORDER for r in con.execute(f"PRAGMA table_info({t})")}
    assert "full_name" not in cols and "phone" not in cols


# ---- Schema: กุญแจและกฎ ----
def test_foreign_key_rejects_unknown_product(con):
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO fact_sales (sale_line_id, receipt_id, date_key, sale_datetime, hour, branch_id, "
                    "product_id, qty, unit_price, line_total, items_in_bill, time_suspect) "
                    "VALUES (999999, 'X', 20260101, '2026-01-01T08:00:00+07:00', 8, 'nimman', 'P99', 1, 55, 55, 1, 0)")


@pytest.mark.parametrize("column, bad", [("qty", 0), ("qty", -1), ("unit_price", 0), ("size", "XL")])
def test_check_constraints_reject_bad_values(con, column, bad):
    row = {"sale_line_id": 999999, "receipt_id": "X", "date_key": 20260101,
           "sale_datetime": "2026-01-01T08:00:00+07:00", "hour": 8, "branch_id": "nimman", "product_id": "P03",
           "size": "M", "qty": 1, "unit_price": 65, "line_total": 65, "items_in_bill": 1, "time_suspect": 0}
    row[column] = bad
    cols = ", ".join(row)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(f"INSERT INTO fact_sales ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))


def test_duplicate_line_rejected(con):
    r = con.execute("SELECT receipt_id, product_id FROM fact_sales LIMIT 1").fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO fact_sales (sale_line_id, receipt_id, date_key, sale_datetime, hour, branch_id, "
                    "product_id, qty, unit_price, line_total, items_in_bill, time_suspect) "
                    "VALUES (999999, ?, 20260101, '2026-01-01T08:00:00+07:00', 8, 'nimman', ?, 1, 55, 55, 1, 0)", r)


# ---- Transaction: ล้มเหลวต้องย้อนกลับทั้งหมด ----
def test_failed_load_rolls_back_everything(tables, tmp_path):
    db = tmp_path / "rollback.db"
    load.run_load(transformed=tables, db_path=db)
    con = load.connect(db)
    bad = dict(tables)
    bad_fact = tables["fact_sales"].copy()
    bad_fact.loc[bad_fact.index[-1], "qty"] = -5          # แถวสุดท้ายผิดกฎ CHECK
    bad["fact_sales"] = bad_fact
    with pytest.raises(sqlite3.IntegrityError):
        load.full_refresh(con, bad)
    # ข้อมูลเดิมต้องยังอยู่ครบ (ไม่ถูกลบ และไม่มีการโหลดครึ่งเดียว)
    for t in load.LOAD_ORDER:
        assert con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == len(tables[t]), t
    last = con.execute("SELECT status FROM etl_load_log ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    assert last == "failed"
    con.close()


# ---- Incremental loading ----
def test_incremental_equals_full_and_is_idempotent(tables, tmp_path):
    demo = load.demo_incremental(tables, tmp_path / "inc.db")
    assert demo.iloc[-1]["identical_to_full"]
    assert demo.iloc[2]["rows_inserted"] == 0          # รันซ้ำต้องไม่เพิ่มแถว
    assert demo.iloc[1]["rows_in_db"] == len(tables["fact_sales"])


def test_load_log_records_runs(loaded):
    log = load.query("SELECT * FROM etl_load_log", loaded[1])
    assert set(log["table_name"]) >= set(load.LOAD_ORDER)
    assert (log["status"] == "success").all()


def test_sample_queries_run(loaded):
    for title, df in loaded[0]["samples"].items():
        assert len(df) > 0, title


def test_read_helpers_close_connection(tables, tmp_path):
    """read_table / query ต้องปิดการเชื่อมต่อจริง (บน Windows ถ้ายังเปิดอยู่จะลบไฟล์ไม่ได้)"""
    db = tmp_path / "close.db"
    load.run_load(transformed=tables, db_path=db)
    load.read_table("dim_branch", db)
    load.query("SELECT COUNT(*) FROM fact_sales", db)
    db.unlink()           # ถ้ามีการเชื่อมต่อค้าง จะเกิด PermissionError
    assert not db.exists()
