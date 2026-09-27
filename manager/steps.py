"""
รายละเอียดขั้นตอน ①–⑤ ของ pipeline (ย้ายมาจากหน้าเว็บเดิม webapp/pages/ raw, extract, quality, clean, transform_load)

แต่ละฟังก์ชันแสดง 1 ขั้นในกล่องที่กดเปิด-ปิดได้ของหน้า "อัปโหลดไฟล์"
ใช้ข้อมูลของชุดข้อมูลที่เลือกอยู่ (data.pipeline_state) ไม่มีตรรกะ DE ใหม่ ไม่เขียนตัวเลขตายตัว
"""

from __future__ import annotations

import json
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

from manager import theme
from pipeline import extract, load
from webapp import charts, data


def recolor(fig, colors: list[str]) -> None:
    """ใช้สีของธีมกับกราฟจาก webapp.charts (ชุดข้อมูลละ 1 สี ตามลำดับ)"""
    for trace, color in zip(fig.data, colors):
        trace.marker.color = color


def summary(state: dict) -> dict[str, list[str]]:
    """ข้อความสรุปสั้น ๆ ของแต่ละขั้น (แสดงบนหัวกล่องแม้ยังไม่เปิด)"""
    rep = state["ext"]["report"].set_index("branch").loc["total"]
    cl = state["cleaned"]
    b, a = cl["quality_before"].overall, cl["quality_after"].overall
    return {
        "s1": ["3 รูปแบบไฟล์ (CSV · Excel · JSON)", f"{int(rep['rows_read']):,} แถว"],
        "s2": [f"{int(rep['rows_out']):,} แถวในตารางกลาง",
               "ตรวจยอดผ่าน" if bool(rep["rows_read"] == rep["rows_out"]) else "ตรวจยอดไม่ผ่าน",
               f"ติดธง {int(rep['product_unmatched']):,} แถว"],
        "s3": [f"คะแนนรวม {b['score_avg']:.1%}", f"{len(cl['quality_before'].rules)} กฎ · 6 มิติ"],
        "s4": [f"{a['score_avg']:.1%} หลังทำความสะอาด", f"ตัดออก {len(cl['removed']):,} แถว",
               f"เหลือ {len(cl['sales']):,} แถว"],
        "s5": [f"{len(state['tables']['fact_sales']):,} แถวใน fact_sales"],
    }


# ---------------------------------------------------------------------------
# ① ข้อมูลดิบ
# ---------------------------------------------------------------------------
def raw(state: dict) -> None:
    theme.tech("Ch2 · Data discovery")
    st.markdown("แต่ละสาขาใช้ระบบ POS คนละยี่ห้อ **ข้อมูลเรื่องเดียวกันจึงเขียนคนละแบบ** ต้องแปลงให้เหมือนกันก่อนรวม (ขั้น ②)")
    raw_dir = data.current()["raw_dir"]
    t1, t2, t3, t4 = st.tabs(["นิมมาน · CSV", "มช. · Excel", "ท่าแพ · JSON", "ข้อมูลอื่น"])
    with t1:
        st.caption("วันที่ วว/ดด/ปี พ.ศ. / รหัสสินค้าสั้น / ขนาด S, M, L")
        st.dataframe(extract.read_nimman(raw_dir / extract.FILES["nimman"]).drop(columns="source_row").head(8),
                     hide_index=True)
    with t2:
        st.caption("หัวคอลัมน์ภาษาไทย / วันที่กับเวลาแยกกัน / ชื่อเมนูภาษาไทย / เก็บราคารวม")
        st.dataframe(extract.read_cmu(raw_dir / extract.FILES["cmu"]).drop(columns="source_row").head(8),
                     hide_index=True)
    with t3:
        st.caption("ซ้อนชั้น 1 ก้อน = 1 บิล / เวลาเป็น UTC (ช้ากว่าเวลาไทย 7 ชั่วโมง) / รหัส SKU / ขนาด Small, Medium, Large")
        payload = json.loads((raw_dir / extract.FILES["thaphae"]).read_text(encoding="utf-8"))
        st.json(payload["receipts"][0], expanded=True)
    with t4:
        st.caption("สมาชิก (เบอร์โทรเขียนหลายรูปแบบ บางคนสมัครซ้ำ) / ตารางสินค้า (Master data) / วันหยุด")
        st.dataframe(extract.read_members(raw_dir / "members.csv").head(10), hide_index=True)
        st.dataframe(extract.read_products(raw_dir / "products.csv"), hide_index=True)
    st.markdown("""
| เรื่อง | นิมมาน | มช. | ท่าแพ |
|---|---|---|---|
| วันเวลา | `15/01/2569 08:32` (พ.ศ.) | `2026-01-15` + `08:40` | `2026-01-15T01:32:00Z` (UTC) |
| สินค้า | `C03` | `ลาเต้` | `DB-COF-LAT` |
| ขนาด | `M` | `กลาง` | `Medium` |
| ราคา | ราคาต่อชิ้น | **ราคารวม** | ราคาต่อชิ้น |
| ชำระเงิน | `QR` | `พร้อมเพย์` | `promptpay` |
""")


# ---------------------------------------------------------------------------
# ② Extract + Integration
# ---------------------------------------------------------------------------
def extract_step(state: dict) -> None:
    theme.tech("Ch1 · Extract", "Ch5 · Schema mapping", "Ch2 · JSON → ตาราง", "Ch2 · Translation mapping",
               "Ch2 · แปลงวันเวลา")
    ext = state["ext"]
    staging = ext["sales"]
    total = ext["report"].set_index("branch").loc["total"]
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("**Schema mapping** (คอลัมน์ของแต่ละสาขา → คอลัมน์กลาง)")
        mapping = pd.DataFrame({b: pd.Series({v: k for k, v in m.items()}) for b, m in extract.SCHEMA_MAP.items()})
        mapping.index.name = "คอลัมน์กลาง"
        st.dataframe(mapping.fillna("-"))
    with c2:
        st.success(f"ตรวจยอดผ่าน: อ่านเข้า {int(total['rows_read']):,} = ออก {int(total['rows_out']):,} แถว",
                   icon=":material/check_circle:")
        if int(total["product_unmatched"]):
            st.warning(f"จับคู่สินค้าไม่ได้ {int(total['product_unmatched']):,} แถว ปล่อยว่างและติดธงไว้ให้ขั้น ④ แก้",
                       icon=":material/flag:")
        st.caption("วันเวลา: นิมมาน ลบ 543 ปี / มช. รวมคอลัมน์วันที่กับเวลา / ท่าแพ แปลง UTC เป็นเวลาไทย (+7)")
    st.markdown("**รายงานการ Extract รายสาขา**")
    st.dataframe(ext["report"], hide_index=True)
    st.markdown("**Before / After:** วันเวลา และ Translation mapping (รหัสสินค้า ขนาด ชำระเงิน)")
    sample = staging.groupby("branch").head(2)
    st.dataframe(sample[["branch", "datetime_raw", "sale_datetime", "product_raw", "product_id", "size_raw", "size",
                         "payment_raw", "payment"]].astype(str), hide_index=True)
    flagged = staging[staging["extract_flags"] != ""]
    if len(flagged):
        st.markdown(f"**แถวที่แปลงค่าไม่ได้** ({len(flagged):,} แถว แสดง 50 แถวแรก)")
        st.dataframe(flagged[["branch", "receipt_id", "product_raw", "product_id", "extract_flags", "source_file",
                              "source_row"]].head(50), hide_index=True)


# ---------------------------------------------------------------------------
# ③ Data Quality
# ---------------------------------------------------------------------------
def quality_step(state: dict) -> None:
    theme.tech("Ch3 · 6 มิติ", "Ch3 · Simple ratio", "Ch3 · Weighted average", "Ch3 · Min operation")
    res = state["cleaned"]["quality_before"]
    o = res.overall
    st.markdown("วัดตาราง Staging ด้วยกฎ 6 มิติ คะแนนรายกฎ = 1 − (ผิด ÷ ตรวจ) รวมระดับมิติด้วย Weighted average และ Min operation")
    c = st.columns(4)
    c[0].metric("คะแนนรวม (เฉลี่ย 6 มิติ)", data.pct(o["score_avg"], 2))
    c[1].metric("มิติที่อ่อนที่สุด", o["weakest_dimension"], data.pct(o["score_min"], 2), delta_color="off")
    c[2].metric("Dark data", f"{o['dark_data_rows']:,} แถว", data.pct(o["dark_data_pct"], 2), delta_color="off")
    c[3].metric("Transformation error rate", data.pct(o["transformation_error_rate"], 2))
    dims = res.dimensions.melt(id_vars=["dimension"], value_vars=["score_avg", "score_min"], var_name="วิธีรวม",
                               value_name="คะแนน")
    dims["วิธีรวม"] = dims["วิธีรวม"].map({"score_avg": "Weighted average", "score_min": "Min operation"})
    fig = charts.bar(dims, x="dimension", y="คะแนน", color="วิธีรวม", title="", text_auto=".1%")
    recolor(fig, ["#B9B2A6", theme.PRI])
    fig.add_hline(y=0.98, line_dash="dash", annotation_text="เป้าหมาย 98%")
    fig.update_yaxes(tickformat=".0%")
    theme.plot(theme.style(fig, 320))
    t1, t2, t3, t4 = st.tabs(["คะแนนรายกฎ", "แยกตามสาขา", "แถวที่ผิดกฎ", "Data profiling"])
    t1.dataframe(res.rules.style.format({"score": "{:.2%}"}), hide_index=True)
    with t2:
        st.dataframe(res.by_branch, hide_index=True)
        st.caption("ถ้าปัญหาเกิดสาขาเดียว สาเหตุมักอยู่ที่ระบบหรือคนของสาขานั้น")
    with t3:
        rule = st.selectbox("เลือกกฎ", res.rules["rule_id"], key="st3_rule",
                            format_func=lambda r: f"{r}: " + res.rules.set_index("rule_id").loc[r, "description"])
        st.dataframe(res.details[res.details["rule_id"] == rule].head(200), hide_index=True)
    t4.dataframe(res.profile_sales.astype(str), hide_index=True)


# ---------------------------------------------------------------------------
# ④ Cleaning
# ---------------------------------------------------------------------------
def cleaning_step(state: dict) -> None:
    theme.tech("Ch4 · Listwise deletion", "Ch4 · Z-score", "Ch4 · DBSCAN", "Ch4 · Isolation Forest",
               "Ch4 · Regression imputation")
    cl = state["cleaned"]
    before, after = cl["quality_before"], cl["quality_after"]
    st.markdown("หลักการ: **ใช้ความจริงก่อนใช้สถิติ** / **แก้เฉพาะสิ่งที่มั่นใจ ไม่มั่นใจให้ตัดออกหรือติดธง** / **ทุกการแก้ต้องบันทึก**")
    c = st.columns(3)
    c[0].metric("คะแนนรวม", data.pct(after.overall["score_avg"], 2),
                f"{(after.overall['score_avg'] - before.overall['score_avg']) * 100:+.2f} จุด")
    c[1].metric("แถวขาย", f"{len(cl['sales']):,}", f"ตัดออก {len(cl['removed']):,} แถว", delta_color="off")
    c[2].metric("สมาชิก", f"{len(cl['members']):,}")
    dims = pd.concat([before.dimensions.assign(ช่วง="ก่อน"), after.dimensions.assign(ช่วง="หลัง")])
    fig = charts.bar(dims, x="dimension", y="score_avg", color="ช่วง", title="", text_auto=".1%")
    recolor(fig, ["#B9B2A6", theme.GOOD])
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    theme.plot(theme.style(fig, 300))

    left, right = st.columns(2)
    with left:
        st.markdown("**⭐ เทียบวิธีหา Outlier ของจำนวนแก้ว**")
        st.dataframe(cl["outlier_summary"], hide_index=True)
        st.caption("IQR ใช้ไม่ได้ (Q1 = Q3 = 1 แก้ว) / DBSCAN พลาดค่าพิมพ์ผิดที่ซ้ำกันเป็นกลุ่ม / "
                   "ตัดสินด้วย Z-score + กฎธุรกิจ \"ไม่เกิน 50 แก้วต่อรายการ\"")
    with right:
        st.markdown("**⭐ เทียบวิธีเติมปีเกิดที่ว่าง (Hold-out)**")
        st.dataframe(cl["imputation"], hide_index=True)
    flags = cl["outlier_flags"]
    view = flags.melt(id_vars="qty", value_vars=["zscore", "iqr", "dbscan", "isolation_forest"],
                      var_name="วิธี", value_name="ว่าเป็น outlier")
    view = view[view["qty"] > 1]
    fig = charts.strip(view, x="วิธี", y="qty", color="ว่าเป็น outlier", title="", log_y=True)
    theme.plot(theme.style(fig, 320))
    st.caption("จำนวนแก้ว (แกนลอการิทึม) ที่แต่ละวิธีตัดสินว่าเป็น outlier (แสดงเฉพาะ 2 แก้วขึ้นไป)")

    t1, t2, t3 = st.tabs(["บันทึกการทำความสะอาด", "แถวที่ตัดออก", "ตัวอย่างแถวที่ถูกแก้"])
    t1.dataframe(cl["log"], hide_index=True)
    t2.dataframe(cl["removed"]["removed_reason"].value_counts().rename_axis("เหตุผล").reset_index(name="แถว"),
                 hide_index=True)
    with t3:
        all_flags = sorted({f for s in cl["sales"]["clean_flags"] for f in s.split(";") if f})
        if all_flags:
            flag = st.selectbox("เลือกประเภทการแก้", all_flags, key="st4_flag")
            cols = ["branch", "receipt_id", "product_raw", "product_id", "size", "qty", "unit_price", "line_total",
                    "member_id", "clean_flags"]
            st.dataframe(cl["sales"][cl["sales"]["clean_flags"].str.contains(flag, regex=False)][cols].head(30),
                         hide_index=True)


# ---------------------------------------------------------------------------
# ⑤ Transform + Load
# ---------------------------------------------------------------------------
def load_step(state: dict) -> None:
    theme.tech("Ch2 · Anonymization", "Ch2 · Discretization", "Ch1 · Full refresh", "Ch1 · Incremental load",
               "Ch1 · Load verification")
    tables = state["tables"]
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Star schema ใน SQLite**")
        st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=record, fontname="Helvetica", style=filled, fillcolor="#F1EFEA", color="#E7E4DD"];
  fact [label="{fact_sales|sale_line_id (PK)\\lreceipt_id\\ldate_key (FK)\\lbranch_id (FK)\\lproduct_id (FK)\\lmember_id (FK)\\l}", fillcolor="#F3E9DF"];
  date [label="{dim_date|date_key (PK)\\l}"]; prod [label="{dim_product|product_id (PK)\\l}"];
  br [label="{dim_branch|branch_id (PK)\\l}"]; mem [label="{dim_member|member_id (PK)\\lphone_hash\\l}"];
  fact -> date; fact -> prod; fact -> br; fact -> mem;
}""")
    with right:
        with closing(load.connect(data.current()["db_path"])) as con:
            verification = load.verify_load(con, tables)
        passed = bool(verification["passed"].all())
        (st.success if passed else st.error)(
            f"Load verification ผ่าน {int(verification['passed'].sum())} จาก {len(verification)} ข้อ",
            icon=":material/check_circle:" if passed else ":material/error:")
        st.dataframe(verification.astype({"expected": str, "actual": str}), hide_index=True, height=260)

    t1, t2, t3, t4 = st.tabs(["Anonymization", "Discretization", "ข้อมูลหาย vs ไม่มีลูกค้า", "ประวัติการโหลด"])
    with t1:
        c1, c2 = st.columns(2)
        c1.caption("ก่อน (หลังทำความสะอาด)")
        c1.dataframe(state["cleaned"]["members"][["member_id", "full_name", "phone"]].head(5), hide_index=True)
        c2.caption("หลัง (dim_member ในฐานข้อมูล)")
        c2.dataframe(data.sql("SELECT member_id, phone_hash FROM dim_member LIMIT 5"), hide_index=True)
    with t2:
        c1, c2 = st.columns(2)
        c1.caption("กลุ่มอายุ")
        c1.dataframe(data.sql("SELECT age_group, COUNT(*) members FROM dim_member GROUP BY age_group ORDER BY age_group"),
                     hide_index=True)
        c2.caption("ระดับการใช้จ่าย (Equal-frequency)")
        c2.dataframe(data.sql("""SELECT spend_tier, COUNT(*) members, MIN(total_spent) min_baht, MAX(total_spent) max_baht
                                 FROM dim_member GROUP BY spend_tier ORDER BY min_baht"""), hide_index=True)
    with t3:
        st.dataframe(data.sql("""SELECT branch, date, day_name, is_holiday, ROUND(expected_bills, 1) expected_bills,
                                        p_zero, data_missing FROM agg_daily WHERE expected_bills IS NOT NULL"""), hide_index=True)
        st.caption("โอกาสขายได้ 0 บิล = e^(−λ) (ปัวซง) ต่ำกว่าเกณฑ์ → ข้อมูลหาย (เก็บเป็นค่าว่าง ไม่ใช่ 0)")
    t4.dataframe(data.sql("SELECT * FROM etl_load_log ORDER BY rowid DESC LIMIT 50"), hide_index=True)

    left, right = st.columns(2)
    with left, theme.card("st5_incr"):
        st.markdown("**สาธิต Incremental loading**")
        st.caption("โหลด ม.ค.–พ.ค. ก่อน แล้วโหลด มิ.ย. เพิ่ม และลองโหลด มิ.ย. ซ้ำ (ใช้ฐานข้อมูลชั่วคราว)")
        if st.button("รันการสาธิต", icon=":material/play_arrow:", key="st5_demo"):
            with tempfile.TemporaryDirectory() as tmp:
                st.dataframe(load.demo_incremental(tables, Path(tmp) / "demo.db"), hide_index=True)
    with right, theme.card("st5_sql"):
        st.markdown("**ช่องพิมพ์ SQL (อ่านอย่างเดียว)**")
        examples = {"(พิมพ์เอง)": "SELECT * FROM v_sales_enriched LIMIT 20", **load.SAMPLE_QUERIES}
        pick = st.selectbox("ตัวอย่างคำถาม", list(examples), key="st5_pick")
        query_text = st.text_area("SQL", examples[pick].strip(), height=120, key=f"st5_sql_{pick}")
        if st.button("รัน SQL", key="st5_run"):
            try:
                st.dataframe(data.safe_select(query_text), hide_index=True)
            except Exception as e:
                st.error(f"รันไม่ได้: {e}")
        st.caption(f"รับเฉพาะ SELECT / WITH คำสั่งเดียว แสดงไม่เกิน {data.MAX_SQL_ROWS:,} แถว")


STEPS = [
    ("s1", "① ข้อมูลดิบ", raw),
    ("s2", "② Extract + Integration", extract_step),
    ("s3", "③ Data Quality", quality_step),
    ("s4", "④ Cleaning", cleaning_step),
    ("s5", "⑤ Transform + Load", load_step),
]
