"""หน้าแรก: เรื่องราวของร้าน, แผนภาพ pipeline, ตัวเลขสรุป"""

import streamlit as st

from webapp import data

st.title("☕ Doi Brew: ETL Pipeline รวมข้อมูลยอดขาย 3 สาขา")
st.markdown(
    "ร้านกาแฟ **Doi Brew** (ข้อมูลสมมุติ) มี 3 สาขาในเชียงใหม่ คือ **นิมมาน, มช., ท่าแพ** "
    "แต่ละสาขาใช้ระบบ POS คนละยี่ห้อ ข้อมูลจึงออกมาคนละรูปแบบ (CSV / Excel / JSON) และมีปัญหาคุณภาพข้อมูลหลายแบบ "
    "โปรแกรมนี้รวม ทำความสะอาด และแปลงข้อมูลให้พร้อมวิเคราะห์ ตามขั้นตอน **ETL (Ch1)**"
)

st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1faee", fontname="Tahoma"];
  raw [label="① ข้อมูลดิบ\\nCSV / Excel / JSON"];
  ext [label="② Extract\\n+ Integration"];
  q   [label="③ Data Quality\\n6 มิติ 21 กฎ"];
  cl  [label="④ Cleaning"];
  tr  [label="⑤ Transform"];
  ld  [label="⑤ Load\\nSQLite"];
  db  [label="Dashboard\\nRecommendation\\nForecast", fillcolor="#e9c46a"];
  raw -> ext -> q -> cl -> tr -> ld -> db;
  cl -> q [label="ตรวจซ้ำ", style=dashed];
}
""")

st.info(f"กำลังใช้: **{data.current()['label']}** (เปลี่ยน / รัน pipeline ใหม่ / อัปโหลดไฟล์ ได้ที่แถบด้านซ้าย)")

state = data.pipeline_state()
before = state["cleaned"]["quality_before"].overall
after = state["cleaned"]["quality_after"].overall
kpi = data.sql("SELECT SUM(line_total) revenue, COUNT(DISTINCT receipt_id) bills, COUNT(*) lines FROM fact_sales").iloc[0]
members = data.sql("SELECT COUNT(*) n FROM dim_member").iloc[0]["n"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("ยอดขายรวม (หลังทำความสะอาด)", data.baht(kpi["revenue"]))
c2.metric("จำนวนบิล", f"{int(kpi['bills']):,}")
c3.metric("สมาชิก", f"{int(members):,}")
c4.metric("คะแนนคุณภาพข้อมูล", data.pct(after["score_avg"]),
          delta=f"{(after['score_avg'] - before['score_avg']) * 100:+.1f} จุด (ก่อน {data.pct(before['score_avg'])})")

st.subheader("แต่ละหน้าแสดงอะไร")
st.markdown("""
| หน้า | แสดงอะไร | บทที่เกี่ยวข้อง |
|---|---|---|
| ① ข้อมูลดิบ | ไฟล์ 3 สาขาที่รูปแบบต่างกัน | Ch2 Data discovery |
| ② Extract + Integration | รวม 3 ไฟล์เป็นตารางเดียว, Schema mapping, แปลงวันเวลา | Ch1, Ch2, Ch5 |
| ③ Data Quality | คะแนนคุณภาพ 6 มิติ 21 กฎ | Ch3 |
| ④ Cleaning | ทำความสะอาด, เทียบวิธีตรวจ outlier และวิธีเติมค่าว่าง | Ch4 |
| ⑤ Transform + Load | Star schema, Anonymization, SQLite, ตรวจหลังโหลด, ช่องพิมพ์ SQL | Ch1, Ch2, Ch5 |
| Dashboard | ยอดขาย ช่วงเวลา เมนู ลูกค้า | Ch6 |
| Recommendation | แนะนำเมนูที่มักซื้อคู่กัน | โมเดลโบนัส |
| Forecast | ทำนายยอดขายรายวัน | โมเดลโบนัส |
""")
