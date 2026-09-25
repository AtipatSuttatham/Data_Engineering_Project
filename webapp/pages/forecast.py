"""Forecast: ทำนายยอดขายรายวันต่อสาขา (โมดูล 8)"""

import pandas as pd
import streamlit as st

from webapp import charts, data

st.title("📈 Forecast: ทำนายยอดขายรายวัน")
st.markdown("**Linear Regression** (อธิบายได้ทุกตัวเลข) ใช้ตัวแปรที่ Encoding / Normalization ไว้ในหน้า ⑤ "
            "กัน Data leakage: แบ่งตามเวลา, เลือกโมเดลด้วยเดือนก่อนหน้า, Z-score จากข้อมูลฝึกเท่านั้น")

res = data.forecast_results()
test = res["test"].set_index("method")
chosen = res["chosen"]
best_base = test.loc[["B1_last_week", "B2_branch_dow_mean"], "MAE"].min()

st.subheader(f"ผลทดสอบเดือน มิ.ย. (โมเดลที่เลือก: {chosen})")
c1, c2, c3 = st.columns(3)
c1.metric("โมเดลทายคลาดเฉลี่ย (MAE)", data.baht(test.loc[chosen, "MAE"]))
c2.metric("Baseline ที่ดีที่สุด", data.baht(best_base))
c3.metric("คลาด % ของยอดเฉลี่ย", data.pct(test.loc[chosen, "MAE_pct_of_mean"]))
st.dataframe(res["test"].round(3), hide_index=True)

st.subheader("ทดสอบหลายเดือน (Nested walk-forward)")
wf = res["walk_forward"].copy()
wf["เดือนที่ทดสอบ"] = wf["test_month"].map(data.MONTH_TH)
st.dataframe(wf.round(1), hide_index=True)
wins = int(wf["beats_both_baselines"].sum())
st.info(f"โมเดลชนะ Baseline ทั้งสองแบบ {wins} จาก {len(wf)} เดือน")

st.subheader("ยอดจริงเทียบยอดที่ทาย (มิ.ย.)")
pred = res["predictions"].copy()
branch = st.radio("สาขา", list(data.BRANCH_TH), format_func=data.BRANCH_TH.get, horizontal=True)
p = pred[pred["branch"] == branch].sort_values("date")
st.plotly_chart(charts.actual_vs_predicted(p, {chosen: f"โมเดล {chosen}", "B1_last_week": "Baseline: สัปดาห์ก่อน",
                                               "B2_branch_dow_mean": "Baseline: ค่าเฉลี่ยวันเดียวกัน"},
                                           f"สาขา{data.BRANCH_TH[branch]}"), width="stretch")

st.subheader("วิเคราะห์ความคลาดเคลื่อน (Residual analysis)")
r = res["residuals"]
by = r["by_branch"].assign(สาขา=lambda x: x["branch"].map(data.BRANCH_TH))
c1, c2 = st.columns(2)
with c1:
    err = r["errors"].assign(สาขา=lambda x: x["branch"].map(data.BRANCH_TH))
    st.plotly_chart(charts.histogram(err, "error", "ความคลาดเคลื่อน (ทาย − จริง) แยกสาขา", color="สาขา"),
                    width="stretch")
with c2:
    st.dataframe(by[["สาขา", "MAE", "bias", "mean_revenue"]].round(0), hide_index=True)
    worst = by.loc[by["bias"].abs().idxmax()]
    msg = (f"สาขา{worst['สาขา']} ถูกทาย{'ต่ำ' if worst['bias'] < 0 else 'สูง'}เกินไปมากที่สุด เฉลี่ย "
           f"{abs(worst['bias']):,.0f} บาท/วัน")
    if worst["branch"] == "cmu" and worst["bias"] < 0:
        msg += (" สาเหตุที่วิเคราะห์ได้: มช. เปิดเทอมวันที่ 1 มิ.ย. แต่โมเดลไม่มีข้อมูลปฏิทินการศึกษา "
                "(แนวทางพัฒนา: เพิ่มปฏิทินการศึกษาเป็นแหล่งข้อมูลใหม่)")
    st.warning(msg)

st.subheader("สิ่งที่โมเดลเรียนรู้ (Coefficient, บาท/วัน)")
coef = res["coefficients"]
st.plotly_chart(charts.bar(coef.sort_values("coefficient_baht"), x="coefficient_baht", y="meaning_th",
                           orientation="h", title="ผลของแต่ละปัจจัยต่อยอดขาย", height=520), width="stretch")

st.subheader("ทำนาย 7 วันถัดไป")
nw = data.table("forecast_next_week")
nw["สาขา"] = nw["branch"].map(data.BRANCH_TH)
nw["date"] = pd.to_datetime(nw["date"])
st.plotly_chart(charts.bar(nw, x="date", y="predicted_revenue", color="สาขา", title="ยอดขายที่คาดการณ์ (บาท)"),
                width="stretch")
st.caption("สมมุติว่าไม่มีวันหยุดในช่วงนี้ / ค่าของ มช. น่าจะต่ำกว่าจริงด้วยเหตุผลเดียวกับเดือน มิ.ย.")
