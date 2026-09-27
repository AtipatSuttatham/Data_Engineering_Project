"""คาดการณ์ 7 วันข้างหน้า (โมเดลโบนัส: Linear Regression) + ความแม่นยำย้อนหลังแบบ Walk-forward"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from manager import common, insights, theme
from manager.theme import BRANCH_COLOR, BRANCH_TH, baht
from webapp import data

nxt = data.table("forecast_next_week").copy()
nxt["date"] = pd.to_datetime(nxt["date"])
fc = data.forecast_results()
res = fc["residuals"]["by_branch"].set_index("branch")

common.header("คาดการณ์ 7 วันข้างหน้า",
              f"{common.thai_date(nxt['date'].min())} – {common.thai_date(nxt['date'].max())} · ทำนายยอดขายรายวันแต่ละสาขา")

# ---- ยอดรวม 7 วันของแต่ละสาขา ----
tot = nxt.groupby("branch")["predicted_revenue"].sum()
m = st.columns(3)
for col, b in zip(m, BRANCH_TH):
    col.metric(f"{BRANCH_TH[b]} · รวม 7 วัน", baht(tot.get(b, 0)),
               f"คลาดเฉลี่ย ± {baht(res.loc[b, 'MAE'])} ต่อวัน" if b in res.index else None, delta_color="off", delta_arrow="off")

with theme.card("fc_chart"):
    theme.heading("ยอดขายที่คาดไว้ต่อวัน", "แท่ง = ค่าคาดการณ์ · เส้น = ± ความคลาดเคลื่อนเฉลี่ยของสาขานั้นในเดือนทดสอบ",
                  ("โมเดลโบนัส · Linear Regression", "Ch2 · One-hot encoding", "Ch2 · Z-score"))
    fig = go.Figure()
    for b in BRANCH_TH:
        g = nxt[nxt["branch"] == b]
        mae = float(res.loc[b, "MAE"]) if b in res.index else 0
        fig.add_bar(x=g["date"].map(lambda d: f"{insights.DAY_TH[d.strftime('%a')]} {d.day}"),
                    y=g["predicted_revenue"], name=BRANCH_TH[b], marker_color=BRANCH_COLOR[b],
                    # ช่วงล่างไม่ต่ำกว่า 0 (ยอดขายติดลบไม่ได้)
                    error_y=dict(type="data", array=[mae] * len(g), arrayminus=g["predicted_revenue"].clip(upper=mae),
                                 color=theme.INK, thickness=1.2, width=4))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="บาท", tickformat=",.0f")
    theme.plot(theme.style(fig, 360))

left, right = st.columns([2, 1])
with left, theme.card("fc_wf"):
    theme.heading("ความแม่นยำย้อนหลัง",
                  "คลาดเคลื่อนเฉลี่ยต่อวัน (MAE, บาท) ยิ่งน้อยยิ่งดี · ทดสอบทีละเดือนโดยไม่ใช้ข้อมูลอนาคต")
    wf = fc["walk_forward"].copy()
    wf["เดือนที่ทดสอบ"] = wf["test_month"].map(data.MONTH_TH)
    wf["ผล"] = wf["beats_both_baselines"].map({True: "ดีกว่าวิธีง่ายทั้งสอง", False: "ไม่ดีกว่า"})
    st.dataframe(
        wf[["เดือนที่ทดสอบ", "MAE_B1", "MAE_B2", "MAE_model", "ผล"]], hide_index=True, width="stretch",
        column_config={"MAE_B1": st.column_config.NumberColumn("เดาจากสัปดาห์ก่อน", format="%,.0f"),
                       "MAE_B2": st.column_config.NumberColumn("ค่าเฉลี่ยวันเดียวกัน", format="%,.0f"),
                       "MAE_model": st.column_config.NumberColumn("โมเดล", format="%,.0f")})
    st.caption(f"โมเดลดีกว่าวิธีง่ายทั้งสอง {int(wf['beats_both_baselines'].sum())} จาก {len(wf)} เดือน")
    if st.toggle("วิธีคำนวณ", key="fc_method"):
        st.markdown("- **ตัวแปร:** ยอด 7 วันก่อน และแนวโน้ม (ปรับด้วย Z-score) · วันในสัปดาห์และสาขา (One-hot) · วันหยุด\n"
                    "- **ทดสอบแบบ Nested walk-forward:** แต่ละเดือนเลือกโมเดลด้วยเดือนก่อนหน้า แล้วฝึกใหม่ด้วยทุกเดือนก่อนเดือนที่ทดสอบ "
                    "ไม่มีการตัดสินใจใดใช้ข้อมูลอนาคต\n"
                    "- **วิธีง่ายที่ใช้เทียบ:** เท่ากับสัปดาห์ก่อน / ค่าเฉลี่ยของสาขาในวันเดียวกันของสัปดาห์")
        coef = fc["coefficients"]
        st.dataframe(coef, hide_index=True, width="stretch")

with right, theme.card("fc_limit"):
    theme.heading("ข้อจำกัด")
    worst = res["bias"].idxmin()
    if res.loc[worst, "bias"] < 0:
        st.warning(f"**{BRANCH_TH[worst]} ถูกคาดต่ำไป** เฉลี่ย {baht(abs(res.loc[worst, 'bias']))} ต่อวันในเดือนทดสอบ "
                   f"(ยอดจริงเฉลี่ย {baht(res.loc[worst, 'mean_revenue'])} ต่อวัน) ข้อมูลยังไม่มีปัจจัยที่อธิบายการเปลี่ยนแปลงนี้",
                   icon=":material/warning:")
    st.caption("ใช้เป็นกรอบวางแผน ไม่ใช่ตัวเลขตายตัว · ไม่รวมผลของวันหยุดในอนาคต")
