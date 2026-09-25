"""Recommendation: แนะนำเมนูที่มักซื้อคู่กัน (โมดูล 7)"""

import streamlit as st

from pipeline import recommend
from webapp import charts, data

st.title("🛒 Recommendation: แนะนำเมนูที่มักซื้อคู่กัน")
st.markdown("Market Basket Analysis ด้วย **Association rules** กฎ \"ซื้อ A → มักซื้อ B\" วัดด้วย "
            "**Support** (เกิดบ่อยแค่ไหน), **Confidence** (ซื้อ A แล้วซื้อ B กี่ %), **Lift** (บ่อยกว่าความบังเอิญกี่เท่า)")

names = data.product_names()
rules = data.table("reco_rules")          # กฎที่ผ่านเกณฑ์ อ่านจาก SQLite
res = data.recommend_results()            # ผลการวัด (คำนวณจากฐานข้อมูลเดียวกัน)

st.subheader("ลองใช้งาน: ลูกค้าสั่งอะไร?")
picked = st.multiselect("เมนูในบิล", list(names), default=["P02"], format_func=names.get)
rec = recommend.recommend(picked, rules, res["popular"]) if picked else None
if rec is not None and rec.empty:
    st.info("เลือกครบทุกเมนูแล้ว ไม่มีเมนูเหลือให้แนะนำ")
elif rec is not None:
    cols = st.columns(len(rec))
    for col, (_, r) in zip(cols, rec.iterrows()):
        if r["source"] == "rule":
            col.success(f"**{names[r['product_id']]}**\n\nเพราะสั่ง {names[r['because_of']]}\n\n"
                        f"ซื้อคู่จริง {data.pct(r['confidence'])} / Lift {r['lift']:.2f} เท่า")
        else:
            col.info(f"**{names[r['product_id']]}**\n\nคำแนะนำสำรอง (เมนูขายดี) ไม่มีกฎสำหรับเมนูนี้")

st.subheader(f"กฎที่ผ่านเกณฑ์ ({len(rules)} กฎ)")
st.caption(f"เกณฑ์: Support ≥ {recommend.MIN_SUPPORT:.0%}, Confidence ≥ {recommend.MIN_CONFIDENCE:.0%}, "
           f"Lift ≥ {recommend.MIN_LIFT} / คำนวณจากข้อมูลฝึก ({res['train_bills']:,} บิล)")
st.dataframe(rules[["antecedent_name", "consequent_name", "count_ab", "support", "confidence", "lift"]]
             .style.format({"support": "{:.1%}", "confidence": "{:.1%}", "lift": "{:.2f}"}), hide_index=True)

st.subheader("วัดผล: ซ่อนทีละเมนูในบิลเดือน มิ.ย. แล้วให้ทาย (Leave-one-out)")
ev = res["evaluation"]
m, b = ev.iloc[0], ev.iloc[1]
c1, c2 = st.columns(2)
c1.metric("ทายอันดับ 1 ถูก (Hit rate@1)", data.pct(m["hit_rate@1"]),
          f"Baseline {data.pct(b['hit_rate@1'])}", delta_color="off")
c2.metric(f"อยู่ใน {recommend.TOP_K} อันดับแรก (Hit rate@{recommend.TOP_K})", data.pct(m[f"hit_rate@{recommend.TOP_K}"]),
          f"Baseline {data.pct(b[f'hit_rate@{recommend.TOP_K}'])}", delta_color="off")
long = ev.melt(id_vars="method", value_vars=["hit_rate@1", f"hit_rate@{recommend.TOP_K}"], var_name="ตัววัด")
fig = charts.bar(long, x="ตัววัด", y="value", color="method", title=f"ระบบแนะนำ vs แนะนำเมนูขายดี ({int(m['cases']):,} กรณี)",
                 text_auto=".1%")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.subheader("กฎเชื่อถือได้ไหม (ข้ามเวลา / ข้ามสาขา)")
st.dataframe(res["stability"].assign(antecedent=lambda x: x["antecedent"].map(names),
                                     consequent=lambda x: x["consequent"].map(names)).round(3), hide_index=True)

with st.expander("ถ้าเปลี่ยนเกณฑ์กรองกฎ"):
    st.dataframe(res["sensitivity"].round(3), hide_index=True)
