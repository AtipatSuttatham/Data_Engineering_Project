"""
โมดูล 9: หน้าเว็บ Streamlit ของโปรเจกต์ Doi Brew ETL

ไฟล์นี้ตั้งค่าหน้าเว็บ แถบด้านซ้าย (เลือกชุดข้อมูล / รัน pipeline ใหม่ / อัปโหลดไฟล์) และเมนู 9 หน้า
เนื้อหาแต่ละหน้าอยู่ใน webapp/pages/ / การดึงข้อมูลอยู่ใน webapp/data.py

วิธีรัน:  py -3.13 -m streamlit run app.py   แล้วเปิด http://localhost:8501
"""

import streamlit as st

from webapp import data

st.set_page_config(page_title="Doi Brew ETL", page_icon="☕", layout="wide")

PAGES = {
    "ภาพรวม": [st.Page("webapp/pages/home.py", title="หน้าแรก", icon="🏠", default=True)],
    "ขั้นตอนของ Pipeline": [
        st.Page("webapp/pages/raw.py", title="① ข้อมูลดิบ", icon="📂"),
        st.Page("webapp/pages/extract.py", title="② Extract + Integration", icon="🔗"),
        st.Page("webapp/pages/quality.py", title="③ Data Quality", icon="🩺"),
        st.Page("webapp/pages/clean.py", title="④ Cleaning", icon="🧹"),
        st.Page("webapp/pages/transform_load.py", title="⑤ Transform + Load", icon="🗄️"),
    ],
    "ผลลัพธ์": [
        st.Page("webapp/pages/dashboard.py", title="Dashboard", icon="📊"),
        st.Page("webapp/pages/recommend.py", title="Recommendation", icon="🛒"),
        st.Page("webapp/pages/forecast.py", title="Forecast", icon="📈"),
    ],
}


def sidebar() -> None:
    st.sidebar.title("☕ Doi Brew ETL")

    # เลือกชุดข้อมูล (ข้อมูลที่อัปโหลดจะแสดงเมื่อรัน pipeline กับไฟล์ที่อัปโหลดแล้ว)
    options = {"sample": "ข้อมูลตัวอย่าง"}
    if data.UPLOAD_DB.exists():
        options["upload"] = "ข้อมูลที่อัปโหลด"
    choice = st.sidebar.radio("ชุดข้อมูล", list(options), format_func=options.get,
                              index=list(options).index(st.session_state.get("dataset", "sample"))
                              if st.session_state.get("dataset", "sample") in options else 0)
    st.session_state["dataset"] = choice

    # รัน pipeline ใหม่ทั้งสาย พร้อมแถบความคืบหน้า
    if st.sidebar.button("🔄 รัน pipeline ใหม่", width="stretch"):
        cur = data.current()
        bar = st.sidebar.progress(0.0, text="เริ่ม ...")
        n = len(data.PIPELINE_STEPS)
        data.run_full_pipeline(cur["raw_dir"], cur["db_path"],
                               on_step=lambda i, name: bar.progress(i / n, text=f"{i + 1}/{n} {name}"))
        bar.progress(1.0, text="เสร็จแล้ว ✓")
        st.sidebar.success("รัน pipeline ครบทุกขั้นแล้ว")

    # อัปโหลดไฟล์ใหม่ แล้วรัน pipeline กับไฟล์นั้น (ใช้ฐานข้อมูลแยก ไม่ทับข้อมูลตัวอย่าง)
    with st.sidebar.expander("📤 อัปโหลดไฟล์ใหม่"):
        st.caption("ไฟล์รูปแบบเดียวกับข้อมูลตัวอย่าง และเป็นข้อมูลช่วง ม.ค.-มิ.ย. 2569 "
                   "(กฎคุณภาพและการแบ่งข้อมูลของโมเดลกำหนดช่วงนี้ไว้) ตารางสินค้าและวันหยุดใช้ของเดิม")
        uploads = {
            "nimman": st.file_uploader("นิมมาน (CSV)", type=["csv"], key="up_nimman"),
            "cmu": st.file_uploader("มช. (Excel)", type=["xlsx"], key="up_cmu"),
            "thaphae": st.file_uploader("ท่าแพ (JSON)", type=["json"], key="up_thaphae"),
            "members": st.file_uploader("สมาชิก (CSV)", type=["csv"], key="up_members"),
        }
        if st.button("ตรวจไฟล์และรัน pipeline", width="stretch"):
            files = {k: f.getvalue() for k, f in uploads.items() if f is not None}
            with st.spinner("กำลังตรวจไฟล์และรัน pipeline ..."):
                problems = data.run_upload(files)
            if problems:
                st.session_state["dataset"] = "sample"
                for p in problems:
                    st.error(p)
            else:
                st.session_state["dataset"] = "upload"
                st.rerun()

    st.sidebar.caption(f"กำลังใช้: **{data.current()['label']}**")


sidebar()
data.ensure_db()
st.navigation(PAGES).run()
