"""
โมดูล 6: โหลดข้อมูลเข้าฐานข้อมูล SQLite (Load)

ขั้น L ของ ETL ใน Ch1: Extract -> Transform -> Load -> Data Repository -> Analytics
โมดูล 7, 8, 9 อ่านข้อมูลจากฐานข้อมูลนี้ (data/processed/doibrew.db)

สิ่งที่ทำ
    1. Schema: ชนิดข้อมูล + Primary key + Foreign key + CHECK + Index (Ch1 "Establishing key relationships")
    2. วิธีโหลด 3 แบบ (Ch1): Initial loading / Full refresh / Incremental loading
       ทุกการโหลดอยู่ใน Transaction เดียว ล้มเหลวกลางทางจะย้อนกลับทั้งหมด (Rollback)
    3. Load verification (Ch1): Missing / null values, Server performance, Load failures
    4. ประวัติการโหลด (etl_load_log) สำหรับตรวจย้อนหลัง (Ch3 Data auditing)

วิธีรัน:  py -3.13 -m pipeline.load
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import transform

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "doibrew.db"

# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------
# ตาราง Star schema เขียน DDL เองทั้งหมด เพื่อกำหนดกุญแจและกฎ (CHECK) ได้
STAR_DDL = {
    "dim_date": """
        CREATE TABLE dim_date (
            date_key      INTEGER PRIMARY KEY,
            date          TEXT    NOT NULL UNIQUE,         -- ISO 8601: 2026-01-15
            year          INTEGER NOT NULL,
            month         INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
            day           INTEGER NOT NULL CHECK (day BETWEEN 1 AND 31),
            day_of_week   INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
            day_name      TEXT    NOT NULL,
            is_weekend    INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
            is_holiday    INTEGER NOT NULL CHECK (is_holiday IN (0, 1)),
            holiday_name  TEXT
        )""",
    "dim_branch": """
        CREATE TABLE dim_branch (
            branch_id     TEXT PRIMARY KEY,
            name_th       TEXT NOT NULL,
            name_en       TEXT NOT NULL,
            source_format TEXT NOT NULL
        )""",
    "dim_product": """
        CREATE TABLE dim_product (
            product_id    TEXT PRIMARY KEY,
            name_th       TEXT NOT NULL,
            name_en       TEXT NOT NULL,
            category      TEXT NOT NULL CHECK (category IN ('coffee', 'non_coffee', 'bakery')),
            price_S       INTEGER CHECK (price_S > 0),
            price_M       INTEGER CHECK (price_M > 0),
            price_L       INTEGER CHECK (price_L > 0),
            price_nosize  INTEGER CHECK (price_nosize > 0)
        )""",
    "dim_member": """
        CREATE TABLE dim_member (
            member_id          TEXT PRIMARY KEY,
            phone_hash         TEXT NOT NULL UNIQUE,         -- SHA-256 + Salt (ไม่มีเบอร์จริง)
            gender             TEXT NOT NULL CHECK (gender IN ('M', 'F', 'U')),
            birth_year         INTEGER NOT NULL,
            birth_year_imputed INTEGER NOT NULL CHECK (birth_year_imputed IN (0, 1)),
            age                INTEGER CHECK (age BETWEEN 10 AND 100),
            age_group          TEXT,
            age_group_ordinal  INTEGER,
            home_branch        TEXT NOT NULL REFERENCES dim_branch (branch_id),
            register_date      TEXT NOT NULL,
            total_spent        REAL NOT NULL CHECK (total_spent >= 0),
            n_bills            INTEGER NOT NULL CHECK (n_bills >= 0),
            n_items            INTEGER NOT NULL CHECK (n_items >= 0),
            spend_tier         TEXT NOT NULL,
            first_purchase     TEXT,
            last_purchase      TEXT,
            favorite_category  TEXT,
            favorite_branch    TEXT REFERENCES dim_branch (branch_id)
        )""",
    "fact_sales": """
        CREATE TABLE fact_sales (
            sale_line_id   INTEGER PRIMARY KEY,
            receipt_id     TEXT    NOT NULL,
            date_key       INTEGER NOT NULL REFERENCES dim_date (date_key),
            sale_datetime  TEXT    NOT NULL,                 -- ISO 8601 พร้อม timezone: 2026-01-15T08:32:00+07:00
            hour           INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
            time_of_day    TEXT,
            branch_id      TEXT    NOT NULL REFERENCES dim_branch (branch_id),
            product_id     TEXT    NOT NULL REFERENCES dim_product (product_id),
            member_id      TEXT    REFERENCES dim_member (member_id),   -- ว่างได้ (ลูกค้าทั่วไป)
            size           TEXT    CHECK (size IN ('S', 'M', 'L')),     -- ว่างได้ (เบเกอรี่)
            size_ordinal   INTEGER CHECK (size_ordinal IN (1, 2, 3)),
            qty            REAL    NOT NULL CHECK (qty > 0),
            unit_price     REAL    NOT NULL CHECK (unit_price > 0),
            line_total     REAL    NOT NULL CHECK (line_total > 0),
            items_in_bill  INTEGER NOT NULL CHECK (items_in_bill >= 1),
            time_suspect   INTEGER NOT NULL CHECK (time_suspect IN (0, 1)),
            clean_flags    TEXT,
            UNIQUE (receipt_id, product_id)                  -- บิลเดียวไม่มีเมนูซ้ำ (ทำให้ Incremental รันซ้ำได้)
        )""",
}

# ตารางสรุป (ตารางวิเคราะห์) สร้าง DDL จากชนิดข้อมูลของ DataFrame พร้อม Primary key ที่กำหนด
ANALYTIC_PK = {
    "agg_daily": ["branch", "date"],
    "agg_hourly": ["branch", "hour"],
    "agg_product": ["product_id", "branch", "month"],
    "basket": ["receipt_id"],
    "model_features": ["branch", "date"],
}

INDEXES = [  # Index ที่คอลัมน์ Foreign key ของ fact (ช่วยให้ join / กรองเร็วขึ้น)
    "CREATE INDEX idx_fact_date ON fact_sales (date_key)",
    "CREATE INDEX idx_fact_branch ON fact_sales (branch_id)",
    "CREATE INDEX idx_fact_product ON fact_sales (product_id)",
    "CREATE INDEX idx_fact_member ON fact_sales (member_id)",
]

VIEWS = {
    "v_sales_enriched": """
        CREATE VIEW v_sales_enriched AS
        SELECT f.*, d.date, d.month, d.day_name, d.is_weekend, d.is_holiday, d.holiday_name,
               b.name_th AS branch_name, p.name_th AS product_name, p.category,
               m.gender, m.age_group, m.spend_tier
        FROM fact_sales f
        JOIN dim_date d    ON f.date_key = d.date_key
        JOIN dim_branch b  ON f.branch_id = b.branch_id
        JOIN dim_product p ON f.product_id = p.product_id
        LEFT JOIN dim_member m ON f.member_id = m.member_id""",
}

LOG_DDL = """
    CREATE TABLE IF NOT EXISTS etl_load_log (
        run_id        TEXT NOT NULL,
        table_name    TEXT NOT NULL,
        mode          TEXT NOT NULL,          -- initial / full_refresh / incremental
        rows_before   INTEGER,
        rows_after    INTEGER,
        rows_inserted INTEGER,
        started_at    TEXT NOT NULL,
        duration_ms   REAL,
        status        TEXT NOT NULL,          -- success / failed
        message       TEXT
    )"""

# ลำดับโหลด: dimension ก่อน fact (fact อ้างถึง dimension) / ลำดับลบกลับกัน
DIMENSIONS = ["dim_date", "dim_branch", "dim_product", "dim_member"]
LOAD_ORDER = DIMENSIONS + ["fact_sales"] + list(ANALYTIC_PK)

# ตารางผลลัพธ์ของโมเดล (โมดูล 7, 8) สร้างจากข้อมูลในฐานข้อมูลนี้และมี Foreign key อ้างถึง dimension
# ต้องลบก่อนสร้าง schema ใหม่ ไม่งั้น SQLite ไม่ยอมลบ dimension ที่ยังถูกอ้างถึง
# (หลังโหลดข้อมูลใหม่ ผลลัพธ์ของโมเดลก็ล้าสมัยอยู่แล้ว ต้องรันโมดูล 7, 8 ใหม่)
DERIVED_TABLES = ["reco_rules", "forecast_daily", "forecast_coefficients", "forecast_next_week"]


def _sqlite_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    return "TEXT"


def ddl_from_frame(name: str, df: pd.DataFrame, pk: list[str]) -> str:
    cols = [f'    "{c}" {_sqlite_type(df[c])}{" NOT NULL" if c in pk else ""}' for c in df.columns]
    return f"CREATE TABLE {name} (\n" + ",\n".join(cols) + f",\n    PRIMARY KEY ({', '.join(pk)})\n)"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")   # SQLite ปิดการตรวจ Foreign key ไว้เป็นค่าเริ่มต้น ต้องเปิดเอง
    return con


def create_schema(con: sqlite3.Connection, tables: dict) -> None:
    """สร้างตารางใหม่ทั้งหมด (ตารางประวัติการโหลดเก็บไว้ ไม่ลบ)"""
    for v in VIEWS:
        con.execute(f"DROP VIEW IF EXISTS {v}")
    for t in DERIVED_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    for t in reversed(LOAD_ORDER):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    for t in DIMENSIONS + ["fact_sales"]:
        con.execute(STAR_DDL[t])
    for t, pk in ANALYTIC_PK.items():
        con.execute(ddl_from_frame(t, tables[t], pk))
    for sql in INDEXES:
        con.execute(sql)
    for sql in VIEWS.values():
        con.execute(sql)
    con.execute(LOG_DDL)
    con.commit()


# ---------------------------------------------------------------------------
# 2. แปลงค่าให้เข้ากับ SQLite และโหลด
# ---------------------------------------------------------------------------
def _to_sql_value(v):
    """แปลงค่า Python / pandas เป็นค่าที่ SQLite เก็บได้"""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()            # 2026-01-15T08:32:00+07:00 (เก็บ timezone ไว้)
    if isinstance(v, date):
        return v.isoformat()            # 2026-01-15
    return v


def _rows(df: pd.DataFrame) -> list[tuple]:
    return [tuple(_to_sql_value(v) for v in row) for row in df.itertuples(index=False, name=None)]


def _insert(con: sqlite3.Connection, table: str, df: pd.DataFrame, ignore_duplicates: bool = False) -> int:
    cols = ", ".join(f'"{c}"' for c in df.columns)
    marks = ", ".join("?" * len(df.columns))
    verb = "INSERT OR IGNORE" if ignore_duplicates else "INSERT"
    before = con.total_changes
    con.executemany(f"INSERT INTO {table} ({cols}) VALUES ({marks})".replace("INSERT", verb, 1), _rows(df))
    return con.total_changes - before


def _count(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _log(con: sqlite3.Connection, rows: list[dict]) -> None:
    con.executemany(
        "INSERT INTO etl_load_log VALUES (:run_id, :table_name, :mode, :rows_before, :rows_after, "
        ":rows_inserted, :started_at, :duration_ms, :status, :message)", rows)
    con.commit()


def full_refresh(con: sqlite3.Connection, tables: dict, names: list[str] | None = None,
                 mode: str = "full_refresh") -> pd.DataFrame:
    """
    Full refresh (Ch1): ลบข้อมูลเดิมทั้งตารางแล้วโหลดใหม่ ทุกตารางใน Transaction เดียว
    ถ้าตารางใดล้มเหลว ทุกตารางย้อนกลับเป็นสภาพเดิม (Rollback) แล้วบันทึกว่าล้มเหลว
    """
    names = names or LOAD_ORDER
    run_id, log = uuid.uuid4().hex[:8], []
    try:
        with con:  # Transaction: สำเร็จทั้งหมด (commit) หรือย้อนกลับทั้งหมด (rollback)
            for t in reversed(names):          # ลบตารางลูก (fact) ก่อนตารางแม่ (dimension)
                log.append({"table_name": t, "rows_before": _count(con, t)})
                con.execute(f"DELETE FROM {t}")
            log.reverse()
            for entry in log:
                t = entry["table_name"]
                start = time.perf_counter()
                entry["started_at"] = datetime.now().isoformat(timespec="seconds")
                entry["rows_inserted"] = _insert(con, t, tables[t])
                entry["duration_ms"] = (time.perf_counter() - start) * 1000
                entry["rows_after"] = _count(con, t)
    except sqlite3.Error as e:
        _log(con, [{"run_id": run_id, "table_name": ",".join(names), "mode": mode, "rows_before": None,
                    "rows_after": None, "rows_inserted": 0, "started_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_ms": None, "status": "failed", "message": str(e)}])
        raise
    for entry in log:
        entry.update({"run_id": run_id, "mode": mode, "status": "success", "message": ""})
    _log(con, log)
    return pd.DataFrame(log)


def load_incremental(con: sqlite3.Connection, new_fact: pd.DataFrame) -> dict:
    """
    Incremental loading (Ch1): โหลดเฉพาะรายการขายที่ใหม่กว่าที่มีอยู่ (Watermark = date_key ล่าสุด)
    ใช้ INSERT OR IGNORE คู่กับ UNIQUE(receipt_id, product_id) ทำให้รันซ้ำกี่รอบก็ไม่เกิดข้อมูลซ้ำ (Idempotent)
    """
    watermark = con.execute("SELECT COALESCE(MAX(date_key), 0) FROM fact_sales").fetchone()[0]
    batch = new_fact[new_fact["date_key"] >= watermark]   # >= เผื่อวันล่าสุดที่อาจโหลดไม่ครบ
    run_id = uuid.uuid4().hex[:8]
    before = _count(con, "fact_sales")
    start = time.perf_counter()
    try:
        with con:
            inserted = _insert(con, "fact_sales", batch, ignore_duplicates=True)
    except sqlite3.Error as e:
        _log(con, [{"run_id": run_id, "table_name": "fact_sales", "mode": "incremental", "rows_before": before,
                    "rows_after": before, "rows_inserted": 0, "started_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_ms": None, "status": "failed", "message": str(e)}])
        raise
    entry = {"run_id": run_id, "table_name": "fact_sales", "mode": "incremental", "rows_before": before,
             "rows_after": _count(con, "fact_sales"), "rows_inserted": inserted,
             "started_at": datetime.now().isoformat(timespec="seconds"),
             "duration_ms": (time.perf_counter() - start) * 1000, "status": "success",
             "message": f"watermark date_key={watermark}, ส่งมา {len(batch)} แถว"}
    _log(con, [entry])
    entry["rows_sent"] = len(batch)
    return entry


# ---------------------------------------------------------------------------
# 3. Load verification (Ch1)
# ---------------------------------------------------------------------------
def verify_load(con: sqlite3.Connection, tables: dict) -> pd.DataFrame:
    """ตรวจหลังโหลด คืนตาราง: การตรวจ / หัวข้อใน Ch1 / ค่าที่ควรเป็น / ค่าจริง / ผ่านไหม"""
    checks = []

    def add(check, topic, expected, actual, passed=None):
        checks.append({"check": check, "ch1_topic": topic, "expected": expected, "actual": actual,
                       "passed": bool(expected == actual) if passed is None else bool(passed)})

    # จำนวนแถว
    for t in LOAD_ORDER:
        add(f"จำนวนแถว {t}", "Load failures", len(tables[t]), _count(con, t))
    # ยอดรวม
    rev_src = round(float(tables["fact_sales"]["line_total"].sum()), 2)
    rev_db = round(con.execute("SELECT SUM(line_total) FROM fact_sales").fetchone()[0], 2)
    add("ยอดขายรวม fact_sales", "Load failures", rev_src, rev_db)
    # ความสอดคล้องระหว่างตารางสรุปกับ fact (คำนวณด้วย SQL)
    mismatch = con.execute("""
        SELECT COUNT(*) FROM agg_daily a
        LEFT JOIN (SELECT f.branch_id, d.date, SUM(f.line_total) rev
                   FROM fact_sales f JOIN dim_date d ON f.date_key = d.date_key GROUP BY 1, 2) x
               ON a.branch = x.branch_id AND a.date = x.date
        WHERE a.data_missing = 0 AND ABS(a.revenue - COALESCE(x.rev, 0)) > 0.005""").fetchone()[0]
    add("agg_daily ตรงกับยอดจาก fact (SQL)", "Load failures", 0, mismatch)
    # ค่าว่าง: ต้องเท่ากับก่อนโหลด (ไม่มีค่าหายหรือเพิ่มระหว่างโหลด)
    for t, col in [("fact_sales", "member_id"), ("fact_sales", "size"), ("fact_sales", "time_of_day"),
                   ("agg_daily", "revenue")]:
        src = int(tables[t][col].isna().sum())
        db = con.execute(f"SELECT COUNT(*) FROM {t} WHERE \"{col}\" IS NULL").fetchone()[0]
        add(f"ค่าว่าง {t}.{col}", "Missing or null values", src, db)
    # กุญแจ
    add("Foreign key ที่อ้างถึงสิ่งที่ไม่มี", "Load failures", 0, len(con.execute("PRAGMA foreign_key_check").fetchall()))
    # เวลาอ่านกลับมาต้องไม่เพี้ยน
    back = pd.to_datetime(pd.read_sql("SELECT sale_datetime FROM fact_sales ORDER BY sale_line_id", con)["sale_datetime"],
                          format="ISO8601")
    src_dt = tables["fact_sales"].sort_values("sale_line_id")["sale_datetime"].reset_index(drop=True)
    add("วันเวลาอ่านกลับตรงกับก่อนโหลด (timezone ไม่เพี้ยน)", "Load failures", len(src_dt),
        int((back.reset_index(drop=True) == src_dt).sum()))
    # ข้อมูลส่วนบุคคล
    pii = [f"{t}.{r[1]}" for t in LOAD_ORDER for r in con.execute(f"PRAGMA table_info({t})") if r[1] in ("full_name", "phone")]
    add("ไม่มีคอลัมน์ชื่อหรือเบอร์โทร", "(Anonymization)", 0, len(pii))
    # ประสิทธิภาพ: เวลาโหลดล่าสุด (แสดงผล ไม่มีเกณฑ์ผ่าน / ไม่ผ่าน)
    last = pd.read_sql("SELECT table_name, duration_ms FROM etl_load_log WHERE status = 'success' "
                       "AND run_id = (SELECT run_id FROM etl_load_log WHERE status='success' AND mode != 'incremental' "
                       "ORDER BY rowid DESC LIMIT 1)", con)
    add("เวลาโหลดรวม (ms)", "Server performance", "-", round(float(last["duration_ms"].sum()), 1), passed=True)
    out = pd.DataFrame(checks)
    for c in ["expected", "actual"]:  # แสดงจำนวนเต็มเป็นจำนวนเต็ม (ไม่ใช่ 39.0)
        out[c] = pd.Series([int(v) if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer()
                            else v for v in out[c]], index=out.index, dtype=object)
    return out


# ---------------------------------------------------------------------------
# สาธิต Incremental loading (สำหรับเล่มรายงานและหน้าเว็บ)
# ---------------------------------------------------------------------------
def demo_incremental(tables: dict, db_path: Path) -> pd.DataFrame:
    """
    สมมุติร้านส่งข้อมูลรายเดือน
      รอบ 1 Initial     : โหลด ม.ค.-พ.ค.
      รอบ 2 Incremental : โหลด มิ.ย.
      รอบ 3 รันซ้ำ       : โหลด มิ.ย. ซ้ำโดยไม่ตั้งใจ ต้องไม่เกิดข้อมูลซ้ำ
    แล้วเทียบกับการโหลดทั้งหมดทีเดียว (Full)
    """
    if db_path.exists():
        db_path.unlink()
    con = connect(db_path)
    try:
        create_schema(con, tables)
        fact = tables["fact_sales"]
        month = pd.to_datetime(fact["date_key"].astype(str)).dt.month
        first, june = fact[month <= 5], fact[month == 6]
        init = {**tables, "fact_sales": first}
        full_refresh(con, init, DIMENSIONS + ["fact_sales"], mode="initial")
        rounds = [{"round": "1 Initial (ม.ค.-พ.ค.)", "rows_sent": len(first), "rows_inserted": len(first),
                   "rows_in_db": _count(con, "fact_sales")}]
        for label in ["2 Incremental (มิ.ย.)", "3 รันซ้ำ (มิ.ย. อีกครั้ง)"]:
            e = load_incremental(con, fact)
            rounds.append({"round": label, "rows_sent": e["rows_sent"],
                           "rows_inserted": e["rows_inserted"], "rows_in_db": e["rows_after"]})
        db = pd.read_sql("SELECT * FROM fact_sales ORDER BY sale_line_id", con)
        same = len(db) == len(fact) and (db["sale_line_id"].to_numpy() == fact["sale_line_id"].to_numpy()).all()
        rounds.append({"round": "ตรวจ: เท่ากับโหลดทั้งหมดทีเดียว", "rows_sent": len(fact),
                       "rows_inserted": None, "rows_in_db": len(db), "identical_to_full": bool(same)})
        return pd.DataFrame(rounds)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# SQL ตัวอย่างตอบคำถามธุรกิจ (แสดงประโยชน์ของ Star schema ใช้ใน Dashboard)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = {
    "เมนูขายดี 3 อันดับของแต่ละสาขา": """
        SELECT branch_name, product_name, SUM(qty) AS cups, SUM(line_total) AS revenue
        FROM v_sales_enriched GROUP BY branch_id, product_id
        ORDER BY branch_name, cups DESC""",
    "ลูกค้ากลุ่มอายุไหนใช้จ่ายมากที่สุด (เฉพาะสมาชิก)": """
        SELECT age_group, COUNT(DISTINCT member_id) AS members, SUM(line_total) AS revenue,
               ROUND(SUM(line_total) * 1.0 / COUNT(DISTINCT member_id), 2) AS revenue_per_member
        FROM v_sales_enriched WHERE member_id IS NOT NULL GROUP BY age_group ORDER BY revenue_per_member DESC""",
    "วันหยุดขายดีกว่าวันธรรมดาเท่าไร (ยอดเฉลี่ยต่อวันต่อสาขา)": """
        SELECT CASE WHEN is_holiday = 1 THEN 'วันหยุดนักขัตฤกษ์'
                    WHEN is_weekend = 1 THEN 'เสาร์-อาทิตย์' ELSE 'วันธรรมดา' END AS day_type,
               branch, ROUND(AVG(revenue), 2) AS avg_revenue
        FROM agg_daily WHERE data_missing = 0 GROUP BY day_type, branch ORDER BY branch, day_type""",
    "ยอดขายรายเดือนแยกหมวดสินค้า": """
        SELECT month, category, SUM(line_total) AS revenue
        FROM v_sales_enriched GROUP BY month, category ORDER BY month, category""",
}


def run_sample_queries(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    out = {}
    for title, sql in SAMPLE_QUERIES.items():
        df = pd.read_sql(sql, con)
        if title.startswith("เมนูขายดี"):
            df = df.groupby("branch_name").head(3)
        out[title] = df
    return out


# ---------------------------------------------------------------------------
# ใช้ในโมดูล 7, 8, 9: อ่านข้อมูลจากฐานข้อมูล
# ---------------------------------------------------------------------------
# หมายเหตุ: "with sqlite3.connect() as con" ของ Python ไม่ได้ปิดการเชื่อมต่อ (แค่ commit / rollback)
# จึงใช้ closing() ให้ปิดจริงทุกครั้ง ไม่งั้นหน้าเว็บที่อ่านซ้ำบ่อยจะมีการเชื่อมต่อค้างสะสม
def read_table(name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as con:
        return pd.read_sql(f"SELECT * FROM {name}", con)


def query(sql: str, db_path: Path = DB_PATH, params: tuple = ()) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as con:
        return pd.read_sql(sql, con, params=params)


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
def run_load(transformed: dict | None = None, db_path: Path = DB_PATH) -> dict:
    """สร้าง schema ใหม่ -> Initial load ทุกตาราง -> Load verification"""
    tables = transformed or transform.run_transform(save=False)
    con = connect(db_path)
    try:
        create_schema(con, tables)
        load_log = full_refresh(con, tables, mode="initial")
        verification = verify_load(con, tables)
        samples = run_sample_queries(con)
    finally:
        con.close()
    return {"tables": tables, "load_log": load_log, "verification": verification, "samples": samples,
            "db_path": db_path}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    res = run_load()
    print(f"=== โหลดเข้า {res['db_path'].name} ===")
    print(res["load_log"][["table_name", "rows_before", "rows_inserted", "rows_after", "duration_ms"]]
          .round(1).to_string(index=False))
    print("\n=== Load verification ===")
    print(res["verification"].to_string(index=False))
    print("\nผ่านทุกข้อ:", bool(res["verification"]["passed"].all()))
    demo_db = DB_PATH.with_name("doibrew_incremental_demo.db")
    print("\n=== สาธิต Incremental loading ===")
    print(demo_incremental(res["tables"], demo_db).to_string(index=False))
    for title, df in res["samples"].items():
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
