"""Dashboard: วิเคราะห์ยอดขายจากฐานข้อมูล (Ch6 Data Exploration & Visualization, Lab 8-9)"""

import pandas as pd
import streamlit as st

from webapp import charts, data

st.title("📊 Dashboard (Ch6)")

daily = data.table("agg_daily")
daily["date"] = pd.to_datetime(daily["date"])
daily["สาขา"] = daily["branch"].map(data.BRANCH_TH)

# ตัวกรอง
c1, c2 = st.columns([2, 3])
branches = c1.multiselect("สาขา", list(data.BRANCH_TH), default=list(data.BRANCH_TH), format_func=data.BRANCH_TH.get)
months = c2.select_slider("เดือน", options=sorted(daily["month"].unique()), value=(daily["month"].min(),
                          daily["month"].max()), format_func=data.MONTH_TH.get)
if not branches:
    st.warning("กรุณาเลือกอย่างน้อย 1 สาขา")
    st.stop()
m_lo, m_hi = months
d = daily[daily["branch"].isin(branches) & daily["month"].between(m_lo, m_hi)]
branch_sql = ",".join(f"'{b}'" for b in branches)
where = f"branch_id IN ({branch_sql}) AND month BETWEEN {int(m_lo)} AND {int(m_hi)}"
sales = data.sql(f"SELECT * FROM v_sales_enriched WHERE {where}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("ยอดขาย", data.baht(sales["line_total"].sum()))
k2.metric("จำนวนบิล", f"{sales['receipt_id'].nunique():,}")
k3.metric("ยอดเฉลี่ยต่อบิล", data.baht(sales["line_total"].sum() / max(sales["receipt_id"].nunique(), 1)))
k4.metric("จำนวนแก้ว/ชิ้น", f"{sales['qty'].sum():,.0f}")

st.subheader("ยอดขายรายวัน")
window = st.radio("Moving average", [7, 14, 30], horizontal=True, format_func=lambda w: f"{w} วัน")
st.plotly_chart(charts.line_with_moving_average(d, "date", "revenue", "สาขา", window,
                                                "ยอดขายรายวันและค่าเฉลี่ยเคลื่อนที่ (Moving average, Ch6)"),
                width="stretch")
missing = d[d["data_missing"] == 1]
if len(missing):
    st.caption(f"ช่องว่างในเส้น = วันที่ข้อมูลหาย {len(missing)} สาขา-วัน (ไม่ใส่ 0 เพื่อไม่ให้เข้าใจผิด)")

st.subheader("เทียบ \"รูปแบบ\" ยอดขายของสาขาที่ขนาดต่างกัน (Min-Max normalization, Ch2)")
mm = d.copy()
mm["revenue_minmax_ma"] = mm.groupby("branch")["revenue_minmax"].transform(lambda s: s.rolling(14, min_periods=1).mean())
st.plotly_chart(charts.lines(mm, "date", "revenue_minmax_ma", "สาขา", "ยอดขายปรับเป็น 0-1 แยกสาขา (เฉลี่ย 14 วัน)",
                             "0 = ต่ำสุดของสาขา / 1 = สูงสุดของสาขา"), width="stretch")

c1, c2 = st.columns(2)
with c1:
    heat = sales[sales["time_suspect"] == 0].pivot_table(index="hour", columns="day_name", values="receipt_id",
                                                        aggfunc="nunique")
    heat = heat.reindex(columns=[c for c in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if c in heat.columns])
    st.plotly_chart(charts.heatmap(heat, "จำนวนบิล: ชั่วโมง × วัน (Heatmap)", "วัน", "ชั่วโมง"),
                    width="stretch")
with c2:
    top = sales.groupby("product_name")["qty"].sum().sort_values().tail(10).reset_index()
    st.plotly_chart(charts.bar(top, x="qty", y="product_name", title="เมนูขายดี 10 อันดับ (จำนวนแก้ว/ชิ้น)",
                               orientation="h", height=420), width="stretch")

c1, c2 = st.columns(2)
with c1:
    cat = sales.groupby(["month", "category"])["line_total"].sum().reset_index()
    cat["เดือน"] = cat["month"].map(data.MONTH_TH)
    st.plotly_chart(charts.bar(cat, x="เดือน", y="line_total", color="category", barmode="stack",
                               title="ยอดขายรายเดือนแยกหมวด (Stacked bar)"), width="stretch")
with c2:
    day_type = d.dropna(subset=["revenue"]).assign(
        ประเภทวัน=lambda x: x["is_holiday"].map({1: "วันหยุดนักขัตฤกษ์"}).fillna(
            x["is_weekend"].map({1: "เสาร์-อาทิตย์", 0: "วันธรรมดา"})))
    avg = day_type.groupby(["สาขา", "ประเภทวัน"])["revenue"].mean().reset_index()
    st.plotly_chart(charts.bar(avg, x="ประเภทวัน", y="revenue", color="สาขา",
                               title="ยอดเฉลี่ยต่อวัน: วันหยุด vs วันธรรมดา", text_auto=".0f"), width="stretch")

st.subheader("ลูกค้าสมาชิก")
members = sales[sales["member_id"].notna()]
c1, c2 = st.columns(2)
with c1:
    age = members.groupby("age_group").agg(revenue=("line_total", "sum"), members=("member_id", "nunique"))
    age["ต่อคน"] = age["revenue"] / age["members"]
    st.plotly_chart(charts.bar(age.reset_index(), x="age_group", y="ต่อคน", title="ยอดใช้จ่ายต่อคนตามกลุ่มอายุ",
                               text_auto=".0f"), width="stretch")
with c2:
    tier = data.sql("SELECT spend_tier, gender, COUNT(*) members FROM dim_member GROUP BY 1, 2")
    st.plotly_chart(charts.bar(tier, x="spend_tier", y="members", color="gender", barmode="stack",
                               title="ระดับการใช้จ่าย × เพศ (U = ไม่ระบุ)"), width="stretch")
st.caption("ตัวเลขในหน้านี้อ่านจากฐานข้อมูล SQLite ทั้งหมด (ตาราง agg_daily, dim_member และ View v_sales_enriched)")
