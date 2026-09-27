"""
คำนวณตัวเลขที่หน้าเว็บ Doi Brew Manager แสดง (pandas ล้วน ไม่มี streamlit จึงทดสอบแยกได้)

ทุกฟังก์ชันรับ DataFrame ที่อ่านจาก SQLite แล้วคืนตารางใหม่ ไม่แก้ข้อมูลที่รับเข้ามา
ไม่มีตัวเลขตายตัว: ข้อมูลเปลี่ยน (เช่น อัปโหลดไฟล์ใหม่) ผลเปลี่ยนตาม
"""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

PERIODS = {"7 วันล่าสุด": 7, "28 วันล่าสุด": 28, "ทั้งหมด": None}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_TH = {"Mon": "จ.", "Tue": "อ.", "Wed": "พ.", "Thu": "พฤ.", "Fri": "ศ.", "Sat": "ส.", "Sun": "อา."}


# ---------------------------------------------------------------------------
# ช่วงเวลา (Filtering, Ch2)
# ---------------------------------------------------------------------------
def period_bounds(last_date: pd.Timestamp, days: int | None, first_date: pd.Timestamp) -> dict:
    """ช่วงที่เลือก + ช่วงก่อนหน้าที่ยาวเท่ากัน (ไว้คิดการเปลี่ยนแปลง) ถ้าเลือกทั้งหมดจะไม่มีช่วงก่อนหน้า"""
    if days is None:
        return {"start": first_date, "end": last_date, "prev_start": None, "prev_end": None}
    start = last_date - pd.Timedelta(days=days - 1)
    return {"start": start, "end": last_date,
            "prev_start": start - pd.Timedelta(days=days), "prev_end": start - pd.Timedelta(days=1)}


def between(df: pd.DataFrame, start, end, col: str = "date") -> pd.DataFrame:
    if start is None:
        return df.iloc[0:0]
    return df[(df[col] >= start) & (df[col] <= end)]


def change(cur: float, prev: float | None) -> float | None:
    """อัตราการเปลี่ยนแปลง (สัดส่วน) ถ้าไม่มีช่วงก่อนหน้าหรือเป็น 0 คืน None"""
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    return cur / prev - 1


# ---------------------------------------------------------------------------
# ตัวเลขสรุป (Aggregation, Ch2)
# ---------------------------------------------------------------------------
def kpis(sales: pd.DataFrame) -> dict:
    bills = sales["receipt_id"].nunique()
    member_bills = sales.loc[sales["member_id"].notna(), "receipt_id"].nunique()
    revenue = float(sales["line_total"].sum())
    return {"revenue": revenue, "bills": int(bills), "avg_bill": revenue / bills if bills else 0.0,
            "member_share": member_bills / bills if bills else 0.0, "items": float(sales["qty"].sum())}


def kpi_changes(cur: pd.DataFrame, prev: pd.DataFrame | None) -> dict:
    """ค่าปัจจุบัน + การเปลี่ยนแปลงเทียบช่วงก่อน (member_share เป็นจุดเปอร์เซ็นต์ ไม่ใช่ %)"""
    now = kpis(cur)
    if prev is None or prev.empty:
        return {k: (v, None) for k, v in now.items()}
    before = kpis(prev)
    out = {k: (now[k], change(now[k], before[k])) for k in now}
    out["member_share"] = (now["member_share"], (now["member_share"] - before["member_share"]) * 100)
    return out


def branch_summary(cur: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """อันดับสาขา: ยอดขาย บิล และการเปลี่ยนแปลง"""
    g = cur.groupby("branch_id").agg(revenue=("line_total", "sum"), bills=("receipt_id", "nunique")).reset_index()
    if prev is not None and not prev.empty:
        p = prev.groupby("branch_id")["line_total"].sum().rename("prev_revenue")
        g = g.join(p, on="branch_id")
        g["change"] = g["revenue"] / g["prev_revenue"] - 1
    else:
        g["change"] = pd.NA
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def product_table(cur: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """ยอดขายรายเมนู (รวมชื่อจาก 3 สาขาแล้วด้วย product_id กลาง) + การเปลี่ยนแปลง"""
    g = (cur.groupby(["product_id", "product_name", "category"])
         .agg(qty=("qty", "sum"), revenue=("line_total", "sum")).reset_index())
    if prev is not None and not prev.empty:
        p = prev.groupby("product_id")["line_total"].sum().rename("prev_revenue")
        g = g.join(p, on="product_id")
        g["change"] = g["revenue"] / g["prev_revenue"] - 1
    else:
        g["change"] = float("nan")
    return g.sort_values("qty", ascending=False).reset_index(drop=True)


def hour_day_heatmap(sales: pd.DataFrame, n_weeks: float) -> pd.DataFrame:
    """จำนวนบิลเฉลี่ยต่อสัปดาห์ ชั่วโมง × วัน (ตัดแถวที่เวลาน่าสงสัยออก)"""
    s = sales[sales["time_suspect"] == 0]
    t = s.pivot_table(index="hour", columns="day_name", values="receipt_id", aggfunc="nunique", fill_value=0)
    t = t.reindex(columns=[d for d in DAY_ORDER if d in t.columns]) / max(n_weeks, 1)
    t.columns = [DAY_TH[d] for d in t.columns]
    return t


def day_type_avg(daily: pd.DataFrame) -> pd.DataFrame:
    """ยอดเฉลี่ยต่อวันตามประเภทวัน ใช้ธงวันหยุดที่ได้จากการ Enrichment (Ch2) ไม่นับวันที่ข้อมูลหาย"""
    d = daily[daily["data_missing"] == 0].dropna(subset=["revenue"]).copy()
    d["day_type"] = "วันธรรมดา"
    d.loc[d["is_weekend"] == 1, "day_type"] = "เสาร์–อาทิตย์"
    d.loc[d["is_holiday"] == 1, "day_type"] = "วันหยุดนักขัตฤกษ์"
    order = {"วันธรรมดา": 0, "เสาร์–อาทิตย์": 1, "วันหยุดนักขัตฤกษ์": 2}
    out = d.groupby("day_type").agg(avg_revenue=("revenue", "mean"), days=("revenue", "size")).reset_index()
    return out.sort_values("day_type", key=lambda s: s.map(order)).reset_index(drop=True)


def minmax_pattern(daily: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """ยอดที่ปรับเป็น 0-1 ต่อสาขา (Min-Max, Ch2) แล้วเฉลี่ยเคลื่อนที่ ให้เทียบรูปแบบของสาขาที่ขนาดต่างกันได้"""
    d = daily.sort_values(["branch", "date"]).copy()
    d["pattern"] = d.groupby("branch")["revenue_minmax"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return d[["branch", "date", "pattern"]]


def category_mix(sales: pd.DataFrame) -> pd.DataFrame:
    g = sales.groupby("category")["line_total"].sum()
    return (g / g.sum()).rename("share").reset_index().sort_values("share", ascending=False)


def age_group_spend(members: pd.DataFrame) -> pd.DataFrame:
    """ใช้จ่ายเฉลี่ยต่อคนตามช่วงอายุ (ช่วงอายุมาจาก Discretization ใน Ch2) เรียงตาม ordinal"""
    m = members.dropna(subset=["age_group"])
    g = (m.groupby(["age_group", "age_group_ordinal"])
         .agg(members=("member_id", "size"), spend_per_member=("total_spent", "mean")).reset_index())
    return g.sort_values("age_group_ordinal").reset_index(drop=True)


def member_vs_walkin(sales: pd.DataFrame) -> dict:
    bills = sales.groupby("receipt_id").agg(total=("line_total", "sum"), member=("member_id", "first"))
    is_member = bills["member"].notna()
    return {"member_avg_bill": float(bills.loc[is_member, "total"].mean()) if is_member.any() else 0.0,
            "walkin_avg_bill": float(bills.loc[~is_member, "total"].mean()) if (~is_member).any() else 0.0}


# ---------------------------------------------------------------------------
# ตรวจไฟล์ที่อัปโหลด (Data discovery, Ch2): ดูชื่อคอลัมน์แล้วบอกว่าเป็นไฟล์อะไร
# ---------------------------------------------------------------------------
KIND_TH = {"nimman": "ยอดขาย · นิมมาน", "cmu": "ยอดขาย · มช.", "thaphae": "ยอดขาย · ท่าแพ", "members": "สมาชิก"}
CANDIDATES = {".csv": ["nimman", "members"], ".xlsx": ["cmu"], ".json": ["thaphae"]}


def read_columns(filename: str, content: bytes) -> tuple[list[str], int]:
    """อ่านชื่อคอลัมน์และจำนวนแถว (JSON นับจำนวนบิล) ถ้าอ่านไม่ได้จะ raise ให้ผู้เรียกแจ้งผู้ใช้"""
    ext = filename[filename.rfind("."):].lower()
    if ext == ".csv":
        df = pd.read_csv(BytesIO(content), encoding="utf-8-sig", dtype=str)
        return list(df.columns), len(df)
    if ext == ".xlsx":
        df = pd.read_excel(BytesIO(content), dtype=str)
        return list(df.columns), len(df)
    payload = json.loads(content.decode("utf-8"))
    receipts = payload.get("receipts") if isinstance(payload, dict) else None
    if not receipts:
        raise ValueError('ต้องเป็น JSON ที่มีรายการ "receipts"')
    return list(receipts[0].keys()), len(receipts)


def detect_kind(filename: str, content: bytes, spec: dict[str, tuple[str, list[str]]]) -> dict:
    """
    เดาชนิดไฟล์จากนามสกุล + ชื่อคอลัมน์ เทียบกับคอลัมน์ที่ต้องมีของแต่ละชนิด (spec = webapp.data.UPLOAD_SPEC)
    คืน dict: kind (None ถ้าไม่ตรงชนิดใด), rows, problem (ข้อความภาษาไทย หรือ None)
    """
    ext = filename[filename.rfind("."):].lower() if "." in filename else ""
    if ext not in CANDIDATES:
        return {"kind": None, "rows": None, "problem": "รองรับเฉพาะไฟล์ CSV, Excel (.xlsx) และ JSON"}
    try:
        cols, rows = read_columns(filename, content)
    except Exception as e:  # ไฟล์เสียหรือผิดรูปแบบ: แจ้งผู้ใช้ ไม่ให้หน้าเว็บ error
        return {"kind": None, "rows": None, "problem": f"อ่านไฟล์ไม่ได้ ({type(e).__name__})"}
    best, best_missing = None, None
    for kind in CANDIDATES[ext]:
        missing = [c for c in spec[kind][1] if c not in cols]
        if best_missing is None or len(missing) < len(best_missing):
            best, best_missing = kind, missing
    if best_missing:
        return {"kind": None, "rows": rows,
                "problem": f"คล้ายไฟล์{KIND_TH[best]} แต่ไม่มีคอลัมน์ {', '.join(best_missing)}"}
    return {"kind": best, "rows": rows, "problem": None}
