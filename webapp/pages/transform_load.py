"""⑤ Transform + Load: Star schema, Anonymization, SQLite, Load verification, ช่องพิมพ์ SQL (โมดูล 5-6)"""

import tempfile
from contextlib import closing
from pathlib import Path

import streamlit as st

from pipeline import load
from webapp import data

st.title("⑤ Transform + Load (Ch1, Ch2, Ch5)")

state = data.pipeline_state()
tables = state["tables"]

st.subheader("Star schema ในฐานข้อมูล SQLite")
st.graphviz_chart("""
digraph {
  node [shape=record, fontname="Tahoma", style=filled, fillcolor="#f1faee"];
  fact [label="{fact_sales|sale_line_id (PK)\\lreceipt_id\\ldate_key (FK)\\lbranch_id (FK)\\lproduct_id (FK)\\lmember_id (FK)\\lqty, unit_price, line_total\\l}", fillcolor="#e9c46a"];
  date [label="{dim_date|date_key (PK)\\lmonth, day_name\\lis_holiday\\l}"];
  prod [label="{dim_product|product_id (PK)\\lname_th, category\\lprice_S / M / L\\l}"];
  br [label="{dim_branch|branch_id (PK)\\lname_th\\l}"];
  mem [label="{dim_member|member_id (PK)\\lphone_hash\\lage_group, spend_tier\\l}"];
  fact -> date; fact -> prod; fact -> br; fact -> mem;
}
""")
counts = data.sql("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
counts["rows"] = [int(data.sql(f"SELECT COUNT(*) n FROM {t}").iloc[0]["n"]) for t in counts["name"]]
st.dataframe(counts, hide_index=True)

st.subheader("Anonymization (Ch2): ลบชื่อ + hash เบอร์โทร")
c1, c2 = st.columns(2)
c1.caption("ก่อน (ข้อมูลหลังทำความสะอาด)")
c1.dataframe(state["cleaned"]["members"][["member_id", "full_name", "phone"]].head(5), hide_index=True)
c2.caption("หลัง (dim_member ในฐานข้อมูล)")
c2.dataframe(data.sql("SELECT member_id, phone_hash FROM dim_member LIMIT 5"), hide_index=True)

st.subheader("Discretization (Ch2)")
c1, c2 = st.columns(2)
c1.caption("กลุ่มอายุ (แบ่งตามช่วงชีวิต)")
c1.dataframe(data.sql("SELECT age_group, COUNT(*) members FROM dim_member GROUP BY age_group ORDER BY age_group"),
             hide_index=True)
c2.caption("ระดับการใช้จ่าย (Equal-frequency: แต่ละกลุ่มมีคนเท่ากัน)")
c2.dataframe(data.sql("""SELECT spend_tier, COUNT(*) members, MIN(total_spent) min_baht, MAX(total_spent) max_baht
                         FROM dim_member GROUP BY spend_tier ORDER BY min_baht"""), hide_index=True)

st.subheader("แยก \"ข้อมูลหาย\" ออกจาก \"ไม่มีลูกค้าจริง\" (วันที่ยอดขายเป็น 0)")
zero = data.sql("""SELECT branch, date, day_name, is_holiday, ROUND(expected_bills, 1) expected_bills,
                          p_zero, data_missing FROM agg_daily WHERE expected_bills IS NOT NULL""")
st.dataframe(zero, hide_index=True)
st.caption("โอกาสขายได้ 0 บิล = e^(−λ) (การแจกแจงปัวซง) ต่ำกว่า 1% ÷ จำนวนสาขา-วัน (Bonferroni) → ข้อมูลหาย (เก็บเป็นค่าว่าง)")

st.subheader("Load verification (Ch1)")
with closing(load.connect(data.current()["db_path"])) as con:
    verification = load.verify_load(con, tables)
# คอลัมน์ expected / actual ปนตัวเลขกับข้อความ ("-") แปลงเป็นข้อความก่อนแสดง ไม่ให้ตัวแปลงตารางของ Streamlit ล้มเหลว
st.dataframe(verification.astype({"expected": str, "actual": str}), hide_index=True)
if verification["passed"].all():
    st.success("ผ่านทุกข้อ")
else:
    st.error("มีข้อที่ไม่ผ่าน")

with st.expander("ประวัติการโหลด (etl_load_log)"):
    st.dataframe(data.sql("SELECT * FROM etl_load_log ORDER BY rowid DESC LIMIT 50"), hide_index=True)

st.subheader("สาธิต Incremental loading (Ch1)")
st.caption("สมมุติร้านส่งข้อมูลรายเดือน: โหลด ม.ค.-พ.ค. ก่อน แล้วโหลด มิ.ย. เพิ่ม และลองโหลด มิ.ย. ซ้ำ (ใช้ฐานข้อมูลชั่วคราว)")
if st.button("▶️ รันการสาธิต"):
    with tempfile.TemporaryDirectory() as tmp:
        st.dataframe(load.demo_incremental(tables, Path(tmp) / "demo.db"), hide_index=True)

st.subheader("ช่องพิมพ์ SQL (อ่านอย่างเดียว)")
examples = {"(พิมพ์เอง)": "SELECT * FROM v_sales_enriched LIMIT 20", **load.SAMPLE_QUERIES}
pick = st.selectbox("ตัวอย่างคำถามธุรกิจ", list(examples))
query_text = st.text_area("SQL", examples[pick].strip(), height=160)
if st.button("รัน SQL"):
    try:
        st.dataframe(data.safe_select(query_text), hide_index=True)
    except Exception as e:
        st.error(f"รันไม่ได้: {e}")
st.caption(f"รับเฉพาะ SELECT / WITH, คำสั่งเดียว, แสดงไม่เกิน {data.MAX_SQL_ROWS:,} แถว และเปิดฐานข้อมูลแบบอ่านอย่างเดียว")
