"""
แดชบอร์ด: สาขา (Heatmap, Min-Max, ประเภทวัน, สัดส่วนหมวด) + เมนู (ยอดรายเมนู) + ลูกค้าสมาชิก (ช่วงอายุ, Anonymization)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from manager import common, insights, theme
from manager.theme import BRANCH_COLOR, BRANCH_TH, CATEGORY_COLOR, baht
from webapp import data

s = common.sales()
d = common.daily()
members = data.table("dim_member")

top, c1, c2 = st.columns([3, 1.2, 1.8], vertical_alignment="bottom")
b = common.period_picker(s, "db_period", c1)
branch = common.branch_picker("db_branch", c2, allow_all=False)
with top:
    common.header("แดชบอร์ด", f"วิเคราะห์ตามสาขา เมนู และลูกค้า · {b['label']} ({common.period_text(b)})")
cur = insights.between(s, b["start"], b["end"])
prev = insights.between(s, b["prev_start"], b["prev_end"])


def section(title: str, sub: str) -> None:
    st.write("")
    st.markdown(f"### {title}")
    st.markdown(f'<p class="sub">{sub}</p>', unsafe_allow_html=True)


# =============================================================================
# สาขา
# =============================================================================
section(f"สาขา{BRANCH_TH[branch]}", "ช่วงเวลาที่ขายดี และรูปแบบยอดขายเทียบสาขาอื่น")
sb, db = s[s["branch_id"] == branch], d[d["branch"] == branch]
left, right = st.columns(2)
with left, theme.card("db_heat"):
    # ใช้อย่างน้อย 4 สัปดาห์ ถ้าช่วงสั้นกว่านี้ตัวเลขแต่ละช่องน้อยเกินจนเห็นรูปแบบไม่ชัด
    h_start = min(b["start"], b["end"] - pd.Timedelta(days=27))
    weeks = ((b["end"] - h_start).days + 1) / 7
    theme.heading("ช่วงเวลาที่ลูกค้าเข้าร้าน",
                  f"จำนวนบิลเฉลี่ยต่อสัปดาห์ · {common.period_text({'start': h_start, 'end': b['end']})}",
                  ("Ch6 · Heatmap", "Ch2 · Attribute construction"))
    theme.plot(theme.heatmap(insights.hour_day_heatmap(insights.between(sb, h_start, b["end"]), weeks), height=400))

with right, theme.card("db_minmax"):
    theme.heading("เทียบรูปแบบยอดขาย 3 สาขา", "ปรับยอดแต่ละสาขาเป็น 0–1 (เฉลี่ย 7 วัน) · 0 = ต่ำสุด, 1 = สูงสุดของสาขานั้น",
                  ("Ch2 · Min-Max normalization",))
    theme.plot(theme.branch_lines(insights.minmax_pattern(d), "date", "pattern", height=400))

left, right = st.columns(2)
with left, theme.card("db_daytype"):
    theme.heading("ยอดเฉลี่ยต่อวันตามประเภทวัน", "ทั้งช่วงข้อมูล · ใช้ปฏิทินวันหยุดที่เพิ่มเข้ากับยอดขาย", ("Ch2 · Enrichment", "Ch6 · Bar"))
    dt = insights.day_type_avg(db)
    dt["label"] = dt.apply(lambda r: f"{baht(r.avg_revenue)} · {r.days} วัน", axis=1)
    theme.plot(theme.hbar(dt, "avg_revenue", "day_type", BRANCH_COLOR[branch], "label", height=190))
with right, theme.card("db_mix"):
    theme.heading("สัดส่วนยอดขายตามหมวด", b["label"], ("Ch2 · Aggregation",))
    fig = go.Figure()
    for r in insights.category_mix(insights.between(sb, b["start"], b["end"])).itertuples():
        fig.add_bar(x=[r.share], y=[""], orientation="h", name=f"{r.category} {r.share:.0%}",
                    marker_color=CATEGORY_COLOR.get(r.category), text=f"{r.share:.0%}", textposition="inside")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(visible=False)
    theme.plot(theme.style(fig, 190))

# =============================================================================
# เมนู
# =============================================================================
section("เมนู", "ทุกสาขา · รวมชื่อเมนูจาก 3 สาขาเป็นรหัสเดียวกัน")
with theme.card("db_menu"):
    t = insights.product_table(cur, prev)
    h, tg = st.columns([3, 1], vertical_alignment="bottom")
    with h:
        theme.heading("ยอดขายรายเมนู", f"{len(t)} เมนู · {b['label']}",
                      ("Ch2 · Filtering", "Ch2 · Aggregation", "Ch2 · Translation mapping"))
    show_all = tg.toggle(f"แสดงทั้ง {len(t)} เมนู", key="db_menu_all")
    cats = ["ทั้งหมด"] + list(theme.CATEGORY_TH.values())
    cat = st.segmented_control("หมวด", cats, default="ทั้งหมด", key="db_cat") or "ทั้งหมด"
    if cat != "ทั้งหมด":
        t = t[t["category"] == cat]
    t.insert(0, "อันดับ", range(1, len(t) + 1))
    t = t if show_all else t.head(6)
    st.dataframe(
        t[["อันดับ", "product_name", "category", "qty", "revenue", "change"]], hide_index=True, width="stretch",
        column_config={
            "product_name": "เมนู", "category": "หมวด",
            "qty": st.column_config.NumberColumn("ขาย (แก้ว/ชิ้น)", format="%,.0f"),
            "revenue": st.column_config.NumberColumn("ยอดขาย (บาท)", format="%,.0f"),
            "change": st.column_config.NumberColumn("เทียบช่วงก่อน", format="percent"),
        })

# =============================================================================
# ลูกค้าสมาชิก
# =============================================================================
section("ลูกค้าสมาชิก", f"{len(members):,} คน · ไม่เก็บชื่อและเบอร์โทร ใช้รหัสแทน")
left, right = st.columns(2)
with left, theme.card("db_age"):
    theme.heading("ใช้จ่ายเฉลี่ยต่อคนตามช่วงอายุ", "ทั้งช่วงข้อมูล · แบ่งอายุเป็นช่วง", ("Ch2 · Discretization", "Ch6 · Bar"))
    ag = insights.age_group_spend(members)
    ag["label"] = ag.apply(lambda r: f"{baht(r.spend_per_member)} · {r.members} คน", axis=1)
    ag["group"] = ag["age_group"].str.replace("<=", "≤ ", regex=False) + " ปี"
    top_i = ag["spend_per_member"].idxmax()
    theme.plot(theme.hbar(ag, "spend_per_member", "group", [theme.ACC if i == top_i else "#B9A48E" for i in ag.index],
                          "label", height=260))

with right, theme.card("db_anon"):
    h, tg = st.columns([2.4, 1], vertical_alignment="bottom")
    with h:
        theme.heading("ข้อมูลสมาชิกที่ระบบเก็บ", "ลบชื่อ และแปลงเบอร์โทรเป็นรหัสที่ย้อนกลับไม่ได้", ("Ch2 · Anonymization",))
    if tg.toggle("ตัวอย่าง", value=True, key="db_anon_show"):
        sample = members.head(5).copy()
        sample["phone_hash"] = sample["phone_hash"].str[:6] + "…" + sample["phone_hash"].str[-4:]
        sample["home_branch"] = sample["home_branch"].map(BRANCH_TH)
        st.dataframe(sample[["member_id", "phone_hash", "age_group", "home_branch"]], hide_index=True, width="stretch",
                     column_config={"member_id": "รหัสสมาชิก", "age_group": "ช่วงอายุ", "home_branch": "สาขาประจำ"})
    mv = insights.member_vs_walkin(s)
    k = insights.kpis(s)
    st.caption(f"บิลจากสมาชิก {k['member_share']:.0%} · ยอดเฉลี่ยต่อบิลของสมาชิก {baht(mv['member_avg_bill'], 1)} "
               f"เทียบลูกค้าทั่วไป {baht(mv['walkin_avg_bill'], 1)}")
