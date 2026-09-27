"""ภาพรวม: ตัวเลขสรุป, ยอดขายรายวัน + Moving average, อันดับสาขา, เมนูขายดี"""

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from manager import common, insights, theme
from manager.theme import BRANCH_TH, baht, delta_text
from webapp import data

s = common.sales()
d = common.daily()

top, c1, c2 = st.columns([3, 1.2, 2.2], vertical_alignment="bottom")
with top:
    common.header("ภาพรวม", f"สรุปยอดขายทุกสาขา · ข้อมูลถึง {common.thai_date(s['date'].max())}")
b = common.period_picker(s, "ov_period", c1)
branch = common.branch_picker("ov_branch", c2)

if branch:
    s, d = s[s["branch_id"] == branch], d[d["branch"] == branch]
cur = insights.between(s, b["start"], b["end"])
prev = insights.between(s, b["prev_start"], b["prev_end"])
k = insights.kpi_changes(cur, prev)

# ---- ตัวเลขสรุป ----
note = "เทียบช่วงก่อนหน้าที่ยาวเท่ากัน" if b["prev_start"] is not None else None
m = st.columns(4)
m[0].metric("ยอดขาย", baht(k["revenue"][0]), delta_text(k["revenue"][1]), help=note)
m[1].metric("จำนวนบิล", f"{k['bills'][0]:,}", delta_text(k["bills"][1]), help=note)
m[2].metric("ยอดเฉลี่ยต่อบิล", baht(k["avg_bill"][0], 1), delta_text(k["avg_bill"][1]), help=note)
m[3].metric("บิลจากลูกค้าสมาชิก", f"{k['member_share'][0] * 100:.0f}%", delta_text(k["member_share"][1], " จุด"), help=note)
theme.tech("Ch2 · Aggregation", "Ch2 · Filtering")

# ---- ยอดขายรายวัน + ข้อมูลที่ควรรู้ ----
left, right = st.columns([2, 1])
with left, theme.card("ov_trend"):
    theme.heading("ยอดขายรายวัน", "เส้นจาง = ยอดจริงรายวัน · เส้นเข้ม = ค่าเฉลี่ยเคลื่อนที่",
                  ("Ch6 · Line", "Ch6 · Moving average"))
    window = st.radio("ค่าเฉลี่ย", [7, 14, 30], horizontal=True, format_func=lambda w: f"{w} วัน", key="ov_ma")
    # แสดงอย่างน้อย 8 สัปดาห์ให้เห็นแนวโน้ม (ตัวเลขสรุปด้านบนยังคิดเฉพาะช่วงที่เลือก)
    shown = insights.between(d, min(b["start"], b["end"] - pd.Timedelta(days=55)), b["end"])
    theme.plot(theme.daily_with_ma(shown, window))
    if st.toggle("แสดงตารางรายวัน", key="ov_table"):
        tbl = shown.assign(สาขา=shown["branch"].map(BRANCH_TH)).pivot_table(
            index="date", columns="สาขา", values="revenue", aggfunc="sum").sort_index(ascending=False)
        tbl.index = tbl.index.map(common.thai_date)
        st.dataframe(tbl.head(14), width="stretch", column_config={c: st.column_config.NumberColumn(format="%,.0f")
                                                                     for c in tbl.columns})

with right, theme.card("ov_about"):
    theme.heading("เกี่ยวกับข้อมูลชุดนี้", "สิ่งที่ควรรู้ก่อนอ่านตัวเลข", ("Ch3 · Completeness", "Ch1 · Load"))
    missing = insights.between(d, b["start"], b["end"])
    missing = missing[missing["data_missing"] == 1]
    if len(missing):
        days = ", ".join(f"{BRANCH_TH[r.branch]} {r.date.day}/{r.date.month}" for r in missing.head(4).itertuples())
        more = f" และอีก {len(missing) - 4} วัน" if len(missing) > 4 else ""
        st.error(f"**ข้อมูลหาย {len(missing)} สาขา-วัน:** {days}{more}\n\nกราฟเว้นเป็นช่องว่าง ไม่เติมเป็น 0 "
                 "เพื่อไม่ให้เข้าใจผิดว่าขายไม่ได้", icon=":material/error:")
    else:
        st.success("ไม่มีวันที่ข้อมูลหายในช่วงนี้", icon=":material/check_circle:")
    q = data.pipeline_state()["cleaned"]
    qb, qa = q["quality_before"].overall["score_avg"], q["quality_after"].overall["score_avg"]
    st.success(f"**ความพร้อมของข้อมูล {qa:.1%}** · ขึ้นจาก {qb:.1%} หลังทำความสะอาด", icon=":material/verified:")
    log = data.sql("SELECT mode, started_at, status FROM etl_load_log WHERE status = 'success' ORDER BY rowid DESC LIMIT 1")
    if len(log):
        st.info(f"**โหลดข้อมูลล่าสุด:** {log['started_at'].iloc[0].replace('T', ' ')} ({log['mode'].iloc[0]})\n\n"
                f"ชุดข้อมูล: {data.current()['label']}", icon=":material/database:")
    try:
        st.page_link("manager/pages/upload.py", label="ดูขั้นตอนของ pipeline", icon=":material/arrow_forward:")
    except StreamlitPageNotFoundError:  # เปิดหน้านี้เดี่ยว ๆ (เช่นใน tests) ไม่มีเมนูให้ลิงก์ไป
        pass

# ---- อันดับสาขา + เมนูขายดี ----
left, right = st.columns(2)
with left, theme.card("ov_rank"):
    theme.heading("อันดับสาขา", f"ยอดขาย {b['label']}", ("Ch2 · Aggregation",))
    rank = insights.branch_summary(cur, prev)
    rank["สาขา"] = rank["branch_id"].map(BRANCH_TH)
    st.dataframe(
        rank[["สาขา", "revenue", "change", "bills"]], hide_index=True, width="stretch",
        column_config={
            "revenue": st.column_config.ProgressColumn("ยอดขาย (บาท)", format="%,.0f", min_value=0,
                                                       max_value=float(rank["revenue"].max() or 1)),
            "change": st.column_config.NumberColumn("เทียบช่วงก่อน", format="percent"),
            "bills": st.column_config.NumberColumn("บิล", format="%,d"),
        })

with right, theme.card("ov_top"):
    theme.heading("เมนูขายดี", "5 อันดับแรก (แก้ว/ชิ้น)", ("Ch6 · Bar",))
    top5 = insights.product_table(cur, None).head(5)
    top5["label"] = top5["qty"].map(lambda q: f"{q:,.0f}")
    theme.plot(theme.hbar(top5, "qty", "product_name", [theme.ACC] + ["#D8C3AE"] * 4, "label", height=230))
