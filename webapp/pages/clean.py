"""④ Cleaning: รักษาตามผลตรวจ แล้วตรวจซ้ำ (โมดูล 4)"""

import pandas as pd
import streamlit as st

from webapp import charts, data

st.title("④ Cleaning (Ch4)")
st.markdown("หลักการ: **ใช้ความจริงก่อนใช้สถิติ** / **แก้เฉพาะสิ่งที่มั่นใจ ไม่มั่นใจให้ตัดออกหรือติดธง** / "
            "**ทุกการแก้ต้องบันทึก**")

cl = data.pipeline_state()["cleaned"]
before, after = cl["quality_before"], cl["quality_after"]

st.subheader("ผลลัพธ์: คะแนนคุณภาพก่อน / หลัง (Validation)")
c1, c2, c3 = st.columns(3)
c1.metric("คะแนนรวม", data.pct(after.overall["score_avg"], 2),
          f"{(after.overall['score_avg'] - before.overall['score_avg']) * 100:+.2f} จุด")
c2.metric("แถวขาย", f"{len(cl['sales']):,}", f"ตัดออก {len(cl['removed']):,} แถว", delta_color="off")
c3.metric("สมาชิก", f"{len(cl['members']):,}")
dims = pd.concat([before.dimensions.assign(ช่วง="ก่อน"), after.dimensions.assign(ช่วง="หลัง")])
fig = charts.bar(dims, x="dimension", y="score_avg", color="ช่วง", title="คะแนนรายมิติ ก่อน / หลังทำความสะอาด",
                 text_auto=".1%")
fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
st.plotly_chart(fig, width="stretch")
st.caption("บางมิติไม่ถึง 100% โดยตั้งใจ: ไม่เดาเพศ (U), ข้อมูลที่หายทั้งวันสร้างใหม่ไม่ได้, "
           "เวลาผิดเก็บไว้พร้อมติดธง, ความทันเวลาต้องแก้ที่รอบการส่งข้อมูล")
with st.expander("ตารางเทียบรายกฎ"):
    st.dataframe(cl["quality_compare"], hide_index=True)

st.subheader("บันทึกการทำความสะอาด (Ch4 ขั้น Reporting)")
st.dataframe(cl["log"], hide_index=True)

st.subheader("⭐ เปรียบเทียบวิธีตรวจจับ Outlier ของจำนวนแก้ว")
st.dataframe(cl["outlier_summary"], hide_index=True)
flags = cl["outlier_flags"]
view = flags.melt(id_vars="qty", value_vars=["zscore", "iqr", "dbscan", "isolation_forest"],
                  var_name="วิธี", value_name="ว่าเป็น outlier")
view = view[view["qty"] > 1]
fig = charts.strip(view, x="วิธี", y="qty", color="ว่าเป็น outlier",
                   title="จำนวนแก้ว (แกนลอการิทึม) ที่แต่ละวิธีตัดสินว่าเป็น outlier (แสดงเฉพาะ 2 แก้วขึ้นไป)", log_y=True)
st.plotly_chart(fig, width="stretch")
st.markdown("- **IQR ใช้ไม่ได้:** แถวส่วนใหญ่ซื้อ 1 แก้ว Q1 = Q3 = 1 จึงได้ IQR = 0\n"
            "- **DBSCAN พลาด:** ค่าพิมพ์ผิดซ้ำกันหลายแถวจนรวมเป็นกลุ่มหนาแน่น\n"
            "- **ตัดสินจริง:** Z-score คัดกรอง + กฎธุรกิจ \"ไม่เกิน 50 แก้วต่อรายการ\" แยกพิมพ์ผิดออกจากออเดอร์ใหญ่จริง")

st.subheader("⭐ เปรียบเทียบวิธีเติมปีเกิดที่ว่าง (Hold-out validation)")
imp = cl["imputation"]
st.dataframe(imp, hide_index=True)
fig = charts.bar(imp.dropna(subset=["mae_years"]), x="method", y="mae_years", title="ทายอายุคลาดเฉลี่ย (ปี) ยิ่งต่ำยิ่งดี",
                 text_auto=".2f")
st.plotly_chart(fig, width="stretch")

st.subheader("แถวที่ตัดออก (พร้อมเหตุผล)")
st.dataframe(cl["removed"]["removed_reason"].value_counts().rename_axis("เหตุผล").reset_index(name="แถว"),
             hide_index=True)

st.subheader("ตัวอย่างแถวที่ถูกแก้ (clean_flags)")
all_flags = sorted({f for s in cl["sales"]["clean_flags"] for f in s.split(";") if f})
flag = st.selectbox("เลือกประเภทการแก้", all_flags)
cols = ["branch", "receipt_id", "product_raw", "product_id", "size", "qty", "unit_price", "line_total", "member_id",
        "clean_flags"]
st.dataframe(cl["sales"][cl["sales"]["clean_flags"].str.contains(flag)][cols].head(30), hide_index=True)
