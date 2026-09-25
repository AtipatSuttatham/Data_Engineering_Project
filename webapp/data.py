"""
ส่วนกลางของหน้าเว็บ: ชุดข้อมูลที่ใช้อยู่, การดึงข้อมูลพร้อม cache, การรัน pipeline,
การตรวจไฟล์อัปโหลด และการรัน SQL แบบอ่านอย่างเดียว

หลักการ
    - หน้า "ขั้นตอน" (ข้อมูลดิบ, Extract, Quality, Cleaning) เรียกฟังก์ชันใน pipeline/ แล้วเก็บผลไว้ (cache)
    - หน้า "ผลลัพธ์" (Transform & Load, Dashboard, โมเดล) อ่านจาก SQLite (Load -> Data repository -> Analytics, Ch1)
    - ไม่อ่านไฟล์เฉลย ตามกฎใน CLAUDE.md หัวข้อ 7
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import clean, extract, forecast, load, recommend, transform

ROOT = Path(__file__).resolve().parents[1]
# ตั้งค่าผ่าน environment variable ได้ (ใช้ใน tests เพื่อไม่แตะไฟล์จริง)
DEFAULT_RAW = Path(os.environ.get("DOIBREW_RAW", extract.RAW_DIR))
DEFAULT_DB = Path(os.environ.get("DOIBREW_DB", load.DB_PATH))
UPLOAD_ROOT = Path(os.environ.get("DOIBREW_UPLOAD", ROOT / "data" / "processed" / "upload"))
UPLOAD_RAW = UPLOAD_ROOT / "raw"
UPLOAD_DB = UPLOAD_ROOT / "doibrew_upload.db"

BRANCH_TH = {"nimman": "นิมมาน", "cmu": "มช.", "thaphae": "ท่าแพ"}
MONTH_TH = {1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.", 4: "เม.ย.", 5: "พ.ค.", 6: "มิ.ย.", 7: "ก.ค."}
MAX_SQL_ROWS = 1000


# ---------------------------------------------------------------------------
# ชุดข้อมูลที่ใช้อยู่ (ข้อมูลตัวอย่าง / ข้อมูลที่อัปโหลด)
# ---------------------------------------------------------------------------
def current() -> dict:
    """ชุดข้อมูลที่เลือกอยู่: raw_dir, db_path, label"""
    if st.session_state.get("dataset") == "upload" and UPLOAD_DB.exists():
        return {"raw_dir": UPLOAD_RAW, "db_path": UPLOAD_DB, "label": "ข้อมูลที่อัปโหลด"}
    return {"raw_dir": DEFAULT_RAW, "db_path": DEFAULT_DB, "label": "ข้อมูลตัวอย่าง"}


def db_version(db_path: Path) -> float:
    """เวลาแก้ไขล่าสุดของไฟล์ฐานข้อมูล ใช้เป็นกุญแจของ cache (ฐานข้อมูลเปลี่ยน -> cache หมดอายุ)"""
    return db_path.stat().st_mtime if db_path.exists() else 0.0


# ---------------------------------------------------------------------------
# รัน pipeline ทั้งสาย (โมดูล 2 -> 8)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = ["Extract (โมดูล 2)", "Quality + Cleaning (โมดูล 3-4)", "Transform (โมดูล 5)",
                  "Load เข้า SQLite (โมดูล 6)", "Recommendation (โมดูล 7)", "Forecast (โมดูล 8)"]


def run_full_pipeline(raw_dir: Path, db_path: Path, on_step=None) -> None:
    """รันทุกขั้นตามลำดับ on_step(i, ชื่อขั้น) ใช้แสดงแถบความคืบหน้าบนหน้าเว็บ"""
    step = on_step or (lambda i, name: None)
    step(0, PIPELINE_STEPS[0])
    ext = extract.run_extract(raw_dir=raw_dir, save=False)
    step(1, PIPELINE_STEPS[1])
    cleaned = clean.run_clean(save=False, ext=ext)
    step(2, PIPELINE_STEPS[2])
    tables = transform.run_transform(save=False, cleaned=cleaned)
    step(3, PIPELINE_STEPS[3])
    load.run_load(transformed=tables, db_path=db_path)
    step(4, PIPELINE_STEPS[4])
    recommend.run_recommend(db_path=db_path, save=True)
    step(5, PIPELINE_STEPS[5])
    forecast.run_forecast(db_path=db_path, save=True)
    clear_caches()


def clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def ensure_db() -> bool:
    """ถ้ายังไม่มีฐานข้อมูล (เช่น เพิ่ง clone โปรเจกต์) ให้รัน pipeline ให้อัตโนมัติ 1 ครั้ง"""
    cur = current()
    if cur["db_path"].exists():
        return True
    with st.spinner("ยังไม่มีฐานข้อมูล กำลังรัน pipeline ครั้งแรก (ประมาณ 1 นาที)..."):
        run_full_pipeline(cur["raw_dir"], cur["db_path"])
    return True


# ---------------------------------------------------------------------------
# ผลลัพธ์ระหว่างทาง (cache) สำหรับหน้าขั้นตอน
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังรัน Extract / Quality / Cleaning / Transform ...")
def _pipeline_state(raw_dir: str, version: float) -> dict:
    """
    ใช้ cache_resource (เก็บวัตถุเดิมไว้ ไม่คัดลอกทุกครั้ง) เพื่อประหยัดหน่วยความจำ
    ฟังก์ชันใน pipeline/ ไม่แก้ข้อมูลที่รับเข้ามา (มี tests ตรวจ) จึงใช้วัตถุร่วมกันได้อย่างปลอดภัย
    """
    ext = extract.run_extract(raw_dir=Path(raw_dir), save=False)
    cleaned = clean.run_clean(save=False, ext=ext)
    tables = transform.run_transform(save=False, cleaned=cleaned)
    return {"ext": ext, "cleaned": cleaned, "tables": tables}


def pipeline_state() -> dict:
    cur = current()
    return _pipeline_state(str(cur["raw_dir"]), db_version(cur["db_path"]))


@st.cache_resource(show_spinner="กำลังคำนวณผลของระบบแนะนำ ...")
def _recommend_results(db_path: str, version: float) -> dict:
    return recommend.run_recommend(db_path=Path(db_path), save=False)


def recommend_results() -> dict:
    cur = current()
    return _recommend_results(str(cur["db_path"]), db_version(cur["db_path"]))


@st.cache_resource(show_spinner="กำลังคำนวณผลของโมเดลทำนายยอดขาย ...")
def _forecast_results(db_path: str, version: float) -> dict:
    return forecast.run_forecast(db_path=Path(db_path), save=False)


def forecast_results() -> dict:
    cur = current()
    return _forecast_results(str(cur["db_path"]), db_version(cur["db_path"]))


# ---------------------------------------------------------------------------
# อ่านจาก SQLite (cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=200)
def _sql(sql: str, db_path: str, version: float) -> pd.DataFrame:
    return load.query(sql, Path(db_path))


def sql(query_text: str) -> pd.DataFrame:
    cur = current()
    return _sql(query_text, str(cur["db_path"]), db_version(cur["db_path"]))


def table(name: str) -> pd.DataFrame:
    return sql(f"SELECT * FROM {name}")


def product_names() -> dict:
    return table("dim_product").set_index("product_id")["name_th"].to_dict()


# ---------------------------------------------------------------------------
# SQL แบบอ่านอย่างเดียว (ช่องพิมพ์ SQL บนหน้าเว็บ)
# ---------------------------------------------------------------------------
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
                        re.IGNORECASE)


def safe_select(query_text: str, db_path: Path | None = None) -> pd.DataFrame:
    """
    รัน SQL ที่ผู้ใช้พิมพ์ แบบปลอดภัย 3 ชั้น
      1. รับเฉพาะคำสั่งเดียวที่ขึ้นต้นด้วย SELECT หรือ WITH
      2. ปฏิเสธคำสั่งที่แก้ข้อมูล (INSERT, DELETE, DROP ...)
      3. เปิดฐานข้อมูลแบบอ่านอย่างเดียว (mode=ro) ถึงหลุดสองชั้นแรกก็แก้ข้อมูลไม่ได้
    """
    q = query_text.strip().rstrip(";").strip()
    if not q:
        raise ValueError("กรุณาพิมพ์คำสั่ง SQL")
    if ";" in q:
        raise ValueError("รันได้ครั้งละ 1 คำสั่งเท่านั้น")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise ValueError("รับเฉพาะคำสั่งที่ขึ้นต้นด้วย SELECT หรือ WITH (อ่านข้อมูลอย่างเดียว)")
    if _FORBIDDEN.search(q):
        raise ValueError("คำสั่งนี้มีคำที่ใช้แก้ไขข้อมูล ไม่อนุญาตบนหน้าเว็บ")
    path = (db_path or current()["db_path"]).resolve()
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as con:
        return pd.read_sql(f"SELECT * FROM ({q}) LIMIT {MAX_SQL_ROWS}", con)


# ---------------------------------------------------------------------------
# อัปโหลดไฟล์
# ---------------------------------------------------------------------------
UPLOAD_SPEC = {
    # ชื่อในหน้าเว็บ: (ชื่อไฟล์มาตรฐาน, คอลัมน์ที่ต้องมี)
    "nimman": ("nimman_sales.csv", list(extract.SCHEMA_MAP["nimman"])),
    "cmu": ("cmu_sales.xlsx", list(extract.SCHEMA_MAP["cmu"]) + ["วันที่", "เวลา"]),
    "thaphae": ("thaphae_sales.json", ["receipt_id", "timestamp", "customer", "payment", "items"]),
    "members": ("members.csv", ["member_id", "full_name", "gender", "birth_year", "phone", "register_date",
                                "home_branch"]),
}


def validate_upload(kind: str, content: bytes) -> list[str]:
    """ตรวจไฟล์ก่อนรัน pipeline คืนรายการปัญหา (ภาษาไทย) ถ้าไม่มีปัญหาคืนรายการว่าง"""
    name, required = UPLOAD_SPEC[kind]
    try:
        if name.endswith(".csv"):
            from io import BytesIO
            cols = list(pd.read_csv(BytesIO(content), nrows=5, encoding="utf-8-sig").columns)
        elif name.endswith(".xlsx"):
            from io import BytesIO
            cols = list(pd.read_excel(BytesIO(content), nrows=5).columns)
        else:
            payload = json.loads(content.decode("utf-8"))
            receipts = payload.get("receipts") if isinstance(payload, dict) else None
            if not receipts:
                return [f"{name}: ต้องเป็น JSON ที่มีรายการ \"receipts\""]
            cols = list(receipts[0].keys())
    except Exception as e:  # ไฟล์เสียหรือผิดรูปแบบ: แจ้งผู้ใช้ ไม่ให้หน้าเว็บ error
        return [f"{name}: อ่านไฟล์ไม่ได้ ({type(e).__name__})"]
    missing = [c for c in required if c not in cols]
    return [f"{name}: ไม่มีคอลัมน์ {', '.join(missing)}"] if missing else []


def save_uploads(files: dict[str, bytes]) -> Path:
    """บันทึกไฟล์ที่อัปโหลดด้วยชื่อมาตรฐาน + คัดลอก Master data (สินค้า, วันหยุด) จากข้อมูลตัวอย่าง"""
    if UPLOAD_ROOT.exists():
        shutil.rmtree(UPLOAD_ROOT)
    UPLOAD_RAW.mkdir(parents=True)
    for kind, content in files.items():
        (UPLOAD_RAW / UPLOAD_SPEC[kind][0]).write_bytes(content)
    for master in ["products.csv", "holidays.csv"]:
        shutil.copy(DEFAULT_RAW / master, UPLOAD_RAW / master)
    return UPLOAD_RAW


def run_upload(files: dict[str, bytes]) -> list[str]:
    """
    ตรวจไฟล์ -> บันทึก -> รัน pipeline ลงฐานข้อมูลแยก คืนรายการปัญหา (ว่าง = สำเร็จ)
    ถ้ารันล้มเหลวกลางทาง ลบไฟล์และฐานข้อมูลที่ค้างทิ้ง ไม่ให้เหลือชุดข้อมูลที่ใช้ไม่ได้ให้เลือก
    """
    missing = [k for k in UPLOAD_SPEC if k not in files]
    if missing:
        return [f"ยังไม่ได้อัปโหลด: {', '.join(missing)}"]
    problems = [p for k, content in files.items() for p in validate_upload(k, content)]
    if problems:
        return problems
    try:
        run_full_pipeline(save_uploads(files), UPLOAD_DB)
    except Exception as e:  # ข้อมูลผิดลึกกว่าที่ตรวจได้ (เช่น ช่วงวันที่ต่างจากที่รองรับ)
        clear_caches()
        shutil.rmtree(UPLOAD_ROOT, ignore_errors=True)
        return [f"รัน pipeline ไม่สำเร็จ: {e}"]
    return []


# ---------------------------------------------------------------------------
# รูปแบบตัวเลข
# ---------------------------------------------------------------------------
def baht(x: float) -> str:
    return f"{x:,.0f} บาท"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
