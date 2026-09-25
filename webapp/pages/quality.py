"""③ Data Quality: ตรวจสุขภาพข้อมูลก่อนทำความสะอาด (โมดูล 3)"""

import streamlit as st

from webapp import charts, data

st.title("③ Data Quality (Ch3)")
st.markdown("**ตรวจสุขภาพก่อนรักษา:** วัดคุณภาพของตาราง Staging ด้วยกฎ 21 ข้อ 6 มิติ "
            "คะแนนรายกฎใช้ **Simple ratio** = 1 − (ผิด ÷ ตรวจ) / รวมระดับมิติด้วย **Weighted average** และ **Min operation**")

res = data.pipeline_state()["cleaned"]["quality_before"]
o = res.overall

c1, c2, c3, c4 = st.columns(4)
c1.metric("คะแนนรวม (เฉลี่ย 6 มิติ)", data.pct(o["score_avg"], 2))
c2.metric("มิติที่อ่อนที่สุด (Min)", o["weakest_dimension"], data.pct(o["score_min"], 2), delta_color="off")
c3.metric("Dark data (คิดยอดขายไม่ได้)", f"{o['dark_data_rows']:,} แถว", data.pct(o["dark_data_pct"], 2),
          delta_color="off")
c4.metric("Data transformation error rate", data.pct(o["transformation_error_rate"], 2))

st.subheader("คะแนนรายมิติ (เป้าหมาย 98%)")
dims = res.dimensions.melt(id_vars=["dimension"], value_vars=["score_avg", "score_min"], var_name="วิธีรวม",
                           value_name="คะแนน")
dims["วิธีรวม"] = dims["วิธีรวม"].map({"score_avg": "Weighted average", "score_min": "Min operation"})
fig = charts.bar(dims, x="dimension", y="คะแนน", color="วิธีรวม", title="ค่าเฉลี่ยบอกภาพรวม / ค่าต่ำสุดบอกจุดอ่อน",
                 text_auto=".1%")
fig.add_hline(y=0.98, line_dash="dash", annotation_text="เป้าหมาย 98%")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")
st.dataframe(res.dimensions, hide_index=True)

st.subheader("คะแนนรายกฎ (21 กฎ)")
st.dataframe(res.rules.style.format({"score": "{:.2%}"}), hide_index=True)

st.subheader("Analysis: แยกตามสาขาเพื่อหาสาเหตุ (Ch3 ขั้น 3)")
st.dataframe(res.by_branch, hide_index=True)
st.caption("ถ้าปัญหาเกิดสาขาเดียว สาเหตุมักอยู่ที่ระบบหรือคนของสาขานั้น เช่น ชื่อเมนูสะกดผิด (CS1) เกิดที่ มช. สาขาเดียว")

st.subheader("กรณีศึกษา CP7: วันที่ไม่มียอดขาย")
cp7 = res.details[res.details["rule_id"] == "CP7"]
st.dataframe(cp7[["key", "branch"]], hide_index=True)
st.caption("กฎบอกได้แค่ว่า \"ผิดกฎ\" ต้องวิเคราะห์ต่อจึงรู้ว่าข้อมูลหายจริง หรือไม่มีลูกค้าจริง (ดูผลการแยกในหน้า ⑤)")

st.subheader("ดูแถวที่ผิดกฎ (ย้อนกลับไปหาไฟล์และแถวต้นฉบับได้)")
rule = st.selectbox("เลือกกฎ", res.rules["rule_id"], format_func=lambda r: f"{r}: " +
                    res.rules.set_index("rule_id").loc[r, "description"])
st.dataframe(res.details[res.details["rule_id"] == rule].head(200), hide_index=True)

st.subheader("Data profiling (สำรวจทุกคอลัมน์)")
st.dataframe(res.profile_sales.astype(str), hide_index=True)
prof = res.profile_sales.set_index("column")
st.caption(f"สังเกต: จำนวนต่ำสุด {prof.loc['qty', 'min']:,.0f} / สูงสุด {prof.loc['qty', 'max']:,.0f} แก้ว, "
           f"ราคาต่อชิ้นสูงสุด {prof.loc['unit_price', 'max']:,.0f} บาท, ราคารวมต่ำสุด {prof.loc['line_total', 'min']:,.0f} บาท "
           "... เห็นความผิดปกติตั้งแต่ขั้นสำรวจข้อมูล")
