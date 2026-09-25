"""① ข้อมูลดิบ: ไฟล์ 3 สาขาวางเทียบกัน ให้เห็นปัญหากลุ่ม A (ต่างกันทั้งไฟล์อย่างเป็นระบบ)"""

import json

import streamlit as st

from pipeline import extract
from webapp import data

st.title("① ข้อมูลดิบ (Data discovery, Ch2)")
st.markdown("แต่ละสาขาใช้ระบบ POS คนละยี่ห้อ **ข้อมูลเรื่องเดียวกันจึงเขียนคนละแบบ** ต้องแปลงให้เหมือนกันก่อนรวม (หน้า ②)")

raw_dir = data.current()["raw_dir"]
c1, c2 = st.columns(2)
with c1:
    st.subheader("นิมมาน: CSV")
    st.caption("วันที่ วว/ดด/ปี พ.ศ. / รหัสสินค้าสั้น / ขนาด S, M, L")
    st.dataframe(extract.read_nimman(raw_dir / extract.FILES["nimman"]).drop(columns="source_row").head(8),
                 hide_index=True)
with c2:
    st.subheader("มช.: Excel")
    st.caption("หัวคอลัมน์ภาษาไทย / วันที่กับเวลาแยกกัน / ชื่อเมนูภาษาไทย / เก็บราคารวม")
    st.dataframe(extract.read_cmu(raw_dir / extract.FILES["cmu"]).drop(columns="source_row").head(8), hide_index=True)

st.subheader("ท่าแพ: JSON ซ้อนชั้น (1 ก้อน = 1 บิล)")
st.caption("เวลาเป็น UTC (ช้ากว่าเวลาไทย 7 ชั่วโมง) / รหัส SKU / ขนาด Small, Medium, Large")
payload = json.loads((raw_dir / extract.FILES["thaphae"]).read_text(encoding="utf-8"))
st.json(payload["receipts"][0], expanded=True)

st.subheader("สรุปความแตกต่าง (ปัญหากลุ่ม A)")
st.markdown("""
| เรื่อง | นิมมาน | มช. | ท่าแพ |
|---|---|---|---|
| วันเวลา | `15/01/2569 08:32` (พ.ศ.) | `2026-01-15` + `08:40` | `2026-01-15T01:32:00Z` (UTC) |
| สินค้า | `C03` | `ลาเต้` | `DB-COF-LAT` |
| ขนาด | `M` | `กลาง` | `Medium` |
| ราคา | ราคาต่อชิ้น | **ราคารวม** | ราคาต่อชิ้น |
| ชำระเงิน | `QR` | `พร้อมเพย์` | `promptpay` |
""")

st.subheader("ข้อมูลอื่น")
t1, t2, t3 = st.tabs(["สมาชิก", "ตารางสินค้า (Master data)", "วันหยุด"])
t1.dataframe(extract.read_members(raw_dir / "members.csv").head(20), hide_index=True)
t1.caption("สังเกต: เบอร์โทรเขียนหลายรูปแบบ บางคนสมัครซ้ำ บางช่องว่าง")
t2.dataframe(extract.read_products(raw_dir / "products.csv"), hide_index=True)
t3.dataframe(extract.read_holidays(raw_dir / "holidays.csv"), hide_index=True)
