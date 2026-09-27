"""
อัปโหลดไฟล์: เพิ่มไฟล์จากเครื่อง POS -> ตรวจชนิดไฟล์ -> ประมวลผลและโหลด
และรายละเอียดขั้นตอน ①–⑤ ของ pipeline (กดเปิด-ปิดทีละขั้น หรือเปิดทั้งหมดด้วยสวิตช์)
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from manager import common, importer, insights, steps, theme
from webapp import data

REQUIRED = ["nimman", "cmu", "thaphae", "members"]
FILE_TYPE = {".csv": "CSV", ".xlsx": "Excel", ".json": "JSON"}
state = st.session_state

common.header("อัปโหลดไฟล์", "เพิ่มไฟล์จากเครื่อง POS แล้วดูว่าแต่ละขั้นของ pipeline ทำอะไรกับข้อมูล")
pstate = data.pipeline_state()
cl = pstate["cleaned"]

left, right = st.columns([2, 1])
with left, theme.card("up_files"):
    theme.heading("ไฟล์", "ระบบดูชื่อคอลัมน์แล้วบอกว่าเป็นไฟล์ของสาขาไหน", ("Ch2 · Data discovery",))
    ups = st.file_uploader("ยอดขาย 3 สาขา (CSV · Excel · JSON) และไฟล์สมาชิก (CSV) · ข้อมูลช่วง ม.ค.–มิ.ย. 2569",
                           type=["csv", "xlsx", "json"], accept_multiple_files=True, key="up_uploader")
    found, rows = {}, []
    for f in ups or []:
        content = f.getvalue()
        r = insights.detect_kind(f.name, content, data.UPLOAD_SPEC)
        dup = r["kind"] in found
        if r["kind"] and not dup:
            found[r["kind"]] = content
        status = ("ไฟล์ซ้ำชนิดเดียวกัน (ใช้ไฟล์แรก)" if dup else r["problem"]) or "พร้อม"
        rows.append({"ชนิด": FILE_TYPE.get(Path(f.name).suffix.lower(), "?"), "ไฟล์": f.name,
                     "เป็นข้อมูล": insights.KIND_TH.get(r["kind"], "—"), "แถว": r["rows"],
                     "ตรวจไฟล์": ("✅ " if status == "พร้อม" else "⚠️ ") + status})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"แถว": st.column_config.NumberColumn(format="%,d")})
    missing = [insights.KIND_TH[k] for k in REQUIRED if k not in found]
    if ups and missing:
        st.info(f"ต้องมีครบ 4 ไฟล์ · ยังขาด: {', '.join(missing)}", icon=":material/info:")
    if st.button("ประมวลผลและโหลด", type="primary", icon=":material/play_arrow:", disabled=bool(missing),
                 key="up_run"):
        try:
            with st.status("กำลังประมวลผล ...", expanded=True) as status:
                st.write("อ่านไฟล์ + Schema mapping (②)")
                folder = importer.stage(found, Path(tempfile.mkdtemp(prefix="doibrew_import_")) / "raw")
                st.write("วัดคุณภาพ + ทำความสะอาด (③ ④)")
                checks = importer.run_checks(folder)
                st.write("แปลง + โหลด + ตรวจหลังโหลด + คำนวณโมเดลใหม่ (⑤)")
                result = importer.load_upload(found, checks)
                status.update(label="โหลดเสร็จแล้ว", state="complete")
            state["up_result"] = result
            state["dataset"] = "upload"
            st.rerun()
        except Exception as e:  # ข้อมูลผิดลึกกว่าที่ตรวจได้: แจ้งผู้ใช้ ข้อมูลตัวอย่างไม่ถูกแตะ
            state["dataset"] = "sample"
            st.error(f"โหลดไม่สำเร็จ ระบบลบข้อมูลที่ค้างแล้ว และใช้ข้อมูลตัวอย่างต่อ ({e})")
    if state.get("up_result") is not None and state.get("dataset") == "upload":
        v = state["up_result"]["verification"]
        st.success(f"โหลดข้อมูลที่อัปโหลดแล้ว · Load verification ผ่าน {int(v['passed'].sum())} จาก {len(v)} ข้อ "
                   "· ทุกหน้ากำลังแสดงข้อมูลชุดนี้", icon=":material/check_circle:")

with right, theme.card("up_current"):
    before, after = cl["quality_before"].overall, cl["quality_after"].overall
    s = common.sales()
    log = data.sql("SELECT started_at FROM etl_load_log WHERE status = 'success' ORDER BY rowid DESC LIMIT 1")
    theme.heading("ข้อมูลที่ใช้อยู่", data.current()["label"], ("Ch3 · Weighted average",))
    st.metric("ความพร้อมของข้อมูล", f"{after['score_avg']:.1%}",
              f"{(after['score_avg'] - before['score_avg']) * 100:+.1f} จุด จากก่อนทำความสะอาด")
    theme.kv([("แถวที่ใช้ได้", f"{len(cl['sales']):,}"), ("สมาชิก", f"{len(cl['members']):,}"),
              ("ช่วงข้อมูล", f"{common.thai_date(s['date'].min())} – {common.thai_date(s['date'].max())}"),
              ("โหลดล่าสุด", log["started_at"].iloc[0].replace("T", " ") if len(log) else "—")])

# ---- ขั้นตอน ①–⑤ ----
st.write("")
h, t = st.columns([3, 1], vertical_alignment="bottom")
with h:
    st.markdown("### ขั้นตอนของ pipeline")
    st.markdown('<p class="sub">กดแต่ละขั้นเพื่อดูรายละเอียด</p>', unsafe_allow_html=True)
show_all = t.toggle("แสดงรายละเอียดทั้งหมด", key="up_show_all")
chips = steps.summary(pstate)
for key, title, render in steps.STEPS:
    # หัวกล่องแสดงสรุปสั้น ๆ ของแต่ละขั้นเสมอ เห็นผลได้โดยไม่ต้องเปิด
    with st.expander(f"{title}  :gray[· {' · '.join(chips[key])}]", expanded=show_all):
        render(pstate)
