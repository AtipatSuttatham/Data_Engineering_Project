"""ส่วนที่ทุกหน้าของ Doi Brew Manager ใช้ร่วมกัน: อ่านตารางจาก SQLite และตัวเลือกช่วงเวลา / สาขา"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from manager import insights
from manager.theme import BRANCH_TH, CATEGORY_TH
from webapp import data

SALES_SQL = """
    SELECT date, branch_id, receipt_id, product_id, product_name, category, qty, line_total,
           member_id, hour, day_name, time_suspect
    FROM v_sales_enriched"""


def sales() -> pd.DataFrame:
    """รายการขายที่ผ่านการทำความสะอาดแล้ว (fact_sales + dimension) จากฐานข้อมูลของชุดข้อมูลที่เลือก"""
    s = data.sql(SALES_SQL).copy()
    s["date"] = pd.to_datetime(s["date"])
    s["category"] = s["category"].map(CATEGORY_TH)
    return s


def daily() -> pd.DataFrame:
    d = data.table("agg_daily").copy()
    d["date"] = pd.to_datetime(d["date"])
    return d


def header(title: str, sub: str) -> None:
    st.title(title)
    st.markdown(f'<p class="sub" style="font-size:15px">{sub}</p>', unsafe_allow_html=True)


def period_picker(s: pd.DataFrame, key: str, col=None) -> dict:
    """เลือกช่วงเวลา (Filtering, Ch2) คืนช่วงที่เลือกและช่วงก่อนหน้าที่ยาวเท่ากัน"""
    where = col or st
    label = where.selectbox("ช่วงเวลา", list(insights.PERIODS), key=key)
    b = insights.period_bounds(s["date"].max(), insights.PERIODS[label], s["date"].min())
    b["label"] = label
    return b


def branch_picker(key: str, col=None, allow_all: bool = True) -> str | None:
    """คืนรหัสสาขา หรือ None = ทุกสาขา"""
    where = col or st
    options = (["all"] if allow_all else []) + list(BRANCH_TH)
    names = {"all": "ทุกสาขา", **BRANCH_TH}
    choice = where.segmented_control("สาขา", options, default=options[0], format_func=names.get, key=key)
    choice = choice or options[0]
    return None if choice == "all" else choice


def thai_date(d: pd.Timestamp) -> str:
    """เช่น 30 มิ.ย. 69 (ปี พ.ศ. 2 หลัก)"""
    return f"{d.day} {data.MONTH_TH.get(d.month, d.month)} {(d.year + 543) % 100:02d}"


def period_text(b: dict) -> str:
    return f"{thai_date(b['start'])} – {thai_date(b['end'])}"
