"""
Doi Brew Manager: หน้าเว็บสำหรับใช้งานจริงของร้าน (แยกจากหน้าเว็บนำเสนอ app.py)

ใช้ฐานข้อมูลและฟังก์ชันเดียวกับหน้าเว็บเดิม (webapp/data.py, pipeline/) จัดเป็น 4 หน้า:
อัปโหลดไฟล์ (+ รายละเอียดขั้นตอน ①–⑤ แบบกดเปิด-ปิด) / ภาพรวม / แดชบอร์ด / คาดการณ์
แสดงเฉพาะเทคนิคในขอบเขตของ CLAUDE.md หัวข้อ 3 (ป้ายเทคนิคบนการ์ด เปิด/ปิดได้ที่แถบบนสุด)

วิธีรัน:  python -m streamlit run manager_app.py --server.port 8502 --theme.primaryColor "#1B1916"
          แล้วเปิด http://localhost:8502 (หน้าเว็บเดิมยังรันที่ http://localhost:8501 ได้ตามปกติ)
          --theme.primaryColor ตั้งสีปุ่มและตัวเลือกเป็นสีเข้ม ไม่ใส่ก็ใช้งานได้
"""

from pathlib import Path

import streamlit as st

from manager import theme
from webapp import data

st.set_page_config(page_title="Doi Brew Manager", page_icon="☕", layout="wide")
theme.apply()
HERE = Path(__file__).parent
st.logo(str(HERE / "manager" / "logo.svg"), size="large", icon_image=str(HERE / "manager" / "logo.svg"))

PAGES = [
    st.Page("manager/pages/upload.py", title="อัปโหลดไฟล์", icon=":material/upload:"),
    st.Page("manager/pages/overview.py", title="ภาพรวม", icon=":material/dashboard:", default=True),
    st.Page("manager/pages/dashboard.py", title="แดชบอร์ด", icon=":material/bar_chart:"),
    st.Page("manager/pages/forecast.py", title="คาดการณ์", icon=":material/trending_up:"),
]


def toolbar() -> None:
    """แถบตัวเลือกบนสุดของทุกหน้า: ชุดข้อมูล (session key เดียวกับหน้าเว็บเดิม) + ป้ายเทคนิค"""
    options = {"sample": "ข้อมูลตัวอย่าง"}
    if data.UPLOAD_DB.exists():
        options["upload"] = "ข้อมูลที่อัปโหลด"
    current = st.session_state.get("dataset", "sample")
    _, c1, c2 = st.columns([5, 1.6, 1.9], vertical_alignment="center")
    choice = c1.selectbox("ชุดข้อมูล", list(options), format_func=options.get, label_visibility="collapsed",
                          index=list(options).index(current) if current in options else 0)
    st.session_state["dataset"] = choice
    st.session_state.setdefault("show_tech", True)
    c2.toggle("ป้ายเทคนิค (โหมดนำเสนอ)", key="show_tech")


nav = st.navigation(PAGES, position="top")
toolbar()
data.ensure_db()
nav.run()
