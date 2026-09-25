"""② Extract + Integration: รวม 3 สาขาเป็นตาราง staging เดียว (โมดูล 2)"""

import pandas as pd
import streamlit as st

from pipeline import extract
from webapp import data

st.title("② Extract + Integration (Ch1, Ch2, Ch5)")
st.markdown("อ่านไฟล์ 3 รูปแบบ แล้ว**แปลงให้เป็นโครงสร้างกลางเดียวกัน**ก่อนต่อเป็นตารางเดียว (Staging) "
            "โมดูลนี้แก้เฉพาะปัญหาที่ต่างกันทั้งไฟล์ (กลุ่ม A) ส่วนข้อมูลซ้ำ ค่าว่าง ค่าผิด **ไม่แตะ** ให้หน้า ③ วัดก่อน")

ext = data.pipeline_state()["ext"]
staging = ext["sales"]

st.subheader("รายงานการ Extract + ตรวจยอด")
st.dataframe(ext["report"], hide_index=True)
total = ext["report"].set_index("branch").loc["total"]
st.success(f"อ่านเข้า {int(total['rows_read']):,} แถว = ออก {int(total['rows_out']):,} แถว (ตรวจยอดผ่าน) "
           f"/ จับคู่สินค้าไม่ได้ {int(total['product_unmatched']):,} แถว (ปล่อยว่าง + ติดธง ให้หน้า ④ แก้)")

st.subheader("Schema mapping (แนวคิด Global-as-View, Ch5)")
mapping = pd.DataFrame({b: pd.Series({v: k for k, v in m.items()}) for b, m in extract.SCHEMA_MAP.items()})
mapping.index.name = "คอลัมน์กลาง"
st.dataframe(mapping.fillna("-"))

st.subheader("Before / After: แปลงวันเวลาเป็น ISO 8601 เวลาไทย")
sample = staging.groupby("branch").head(3)[["branch", "datetime_raw", "sale_datetime"]]
st.dataframe(sample.assign(sale_datetime=sample["sale_datetime"].astype(str)), hide_index=True)
st.caption("นิมมาน: ลบ 543 ปี / มช.: รวมคอลัมน์วันที่กับเวลา / ท่าแพ: แปลง UTC เป็นเวลาไทย (+7 ชั่วโมง)")

st.subheader("Before / After: Translation mapping (รหัสสินค้า ขนาด ชำระเงิน)")
st.dataframe(staging.groupby("branch").head(2)[["branch", "product_raw", "product_id", "size_raw", "size",
                                                "payment_raw", "payment"]], hide_index=True)

st.subheader("แถวที่แปลงค่าไม่ได้ (ติดธงใน extract_flags)")
flagged = staging[staging["extract_flags"] != ""]
st.dataframe(flagged[["branch", "receipt_id", "product_raw", "product_id", "extract_flags", "source_file",
                      "source_row"]].head(50), hide_index=True)
st.caption(f"ทั้งหมด {len(flagged):,} แถว (แสดง 50 แถวแรก) ส่วนใหญ่คือชื่อเมนูที่พนักงาน มช. พิมพ์เอง")

with st.expander("ดูตาราง Staging (100 แถวแรก)"):
    st.dataframe(staging.head(100).astype(str), hide_index=True)
