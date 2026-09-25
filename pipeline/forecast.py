"""
โมดูล 8: ทำนายยอดขายรายวันต่อสาขา (Regression)

ใช้ Linear Regression เพราะอธิบายได้ทุกตัวเลข (coefficient) เช่น "วันหยุดทำให้ยอด มช. ลดลง x บาท"
เป็นโมเดลโบนัส (นอกเนื้อหาวิชา) ส่วนที่เชื่อมกับวิชา:
    - ใช้ตัวแปรที่ทำ One-hot / Dummy encoding และ Z-score ไว้ในโมดูล 5 (Ch2, Lab 4)
    - Linear Regression ตัวเดียวกับ Regression imputation ในโมดูล 4 (Ch4)
    - วิเคราะห์ความคลาดเคลื่อนด้วย Line chart / Histogram (Ch6)

กัน Data leakage (ข้อมูลอนาคตรั่วเข้าโมเดล)
    - แบ่งข้อมูลตามเวลา: ฝึก ม.ค.-เม.ย. / เลือกโมเดลด้วย พ.ค. / ทดสอบ มิ.ย. ครั้งเดียว
    - คำนวณ Z-score ใหม่จากช่วงที่ใช้ฝึกเท่านั้น
    - ใช้เฉพาะข้อมูลที่รู้ล่วงหน้า (ยอด 7 วันก่อน, วันในสัปดาห์, วันหยุด, ลำดับวัน)

วิธีรัน:  py -3.13 -m pipeline.forecast
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pipeline import load

BRANCHES = ["cmu", "nimman", "thaphae"]
BRANCH_COLS = [f"branch_{b}" for b in BRANCHES]            # One-hot สาขา (ทำหน้าที่แทนค่าคงที่)
WEEKDAY_COLS = ["dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri"]  # Dummy วันธรรมดา (ฐาน = วันจันทร์)
WEEKEND_COLS = ["dow_Sat", "dow_Sun"]
SCALED = ["lag7_revenue", "trend"]                          # ตัวเลขที่ต้องทำ Z-score

TRAIN_MONTHS = [1, 2, 3, 4]
VALIDATION_MONTH = 5
TEST_MONTH = 6
FORECAST_DAYS = 7    # ทำนายล่วงหน้าได้ 7 วัน เพราะต้องใช้ยอดของ 7 วันก่อน

# ตัวแปรของแต่ละโมเดล
#   M1 พื้นฐาน: สาขา + วันในสัปดาห์ + วันหยุด + ยอดสัปดาห์ก่อน + แนวโน้ม
#   M2 + Interaction: ผลของวันหยุดและเสาร์-อาทิตย์ "แยกตามสาขา" (Attribute construction, Ch2)
#      ออกแบบไม่ให้ซ้ำซ้อน (กัน Dummy variable trap): ใช้วันหยุดแยกสาขา 3 ตัว "แทน" วันหยุดรวม
#      และใช้เสาร์-อาทิตย์แยกสาขา 3 ตัว "แทน" dummy วันเสาร์ / วันอาทิตย์
FEATURES = {
    "M1": BRANCH_COLS + WEEKDAY_COLS + WEEKEND_COLS + ["is_holiday", "lag7_revenue_z", "trend_z"],
    "M2": BRANCH_COLS + WEEKDAY_COLS + [f"weekend_{b}" for b in BRANCHES] + [f"holiday_{b}" for b in BRANCHES]
          + ["lag7_revenue_z", "trend_z"],
}
FEATURE_TH = {
    "branch_cmu": "ยอดตั้งต้น: มช. (วันจันทร์ปกติ)", "branch_nimman": "ยอดตั้งต้น: นิมมาน (วันจันทร์ปกติ)",
    "branch_thaphae": "ยอดตั้งต้น: ท่าแพ (วันจันทร์ปกติ)",
    "dow_Tue": "วันอังคาร (เทียบวันจันทร์)", "dow_Wed": "วันพุธ (เทียบวันจันทร์)",
    "dow_Thu": "วันพฤหัส (เทียบวันจันทร์)", "dow_Fri": "วันศุกร์ (เทียบวันจันทร์)",
    "dow_Sat": "วันเสาร์ (เทียบวันจันทร์)", "dow_Sun": "วันอาทิตย์ (เทียบวันจันทร์)",
    "is_holiday": "วันหยุดนักขัตฤกษ์",
    "weekend_cmu": "เสาร์-อาทิตย์: มช.", "weekend_nimman": "เสาร์-อาทิตย์: นิมมาน",
    "weekend_thaphae": "เสาร์-อาทิตย์: ท่าแพ",
    "holiday_cmu": "วันหยุด: มช.", "holiday_nimman": "วันหยุด: นิมมาน", "holiday_thaphae": "วันหยุด: ท่าแพ",
    "lag7_revenue_z": "ยอดวันเดียวกันของสัปดาห์ก่อน (ต่อ 1 SD)", "trend_z": "แนวโน้มตามเวลา (ต่อ 1 SD)",
}


# ---------------------------------------------------------------------------
# เตรียมข้อมูล
# ---------------------------------------------------------------------------
def load_features(db_path: Path = load.DB_PATH) -> pd.DataFrame:
    """อ่าน model_features จาก SQLite (โมดูล 6) ไม่มีวันที่ข้อมูลหายอยู่แล้ว (ตัดไว้ในโมดูล 5)"""
    df = load.read_table("model_features", db_path)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    return add_interactions(df)


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างตัวแปร วันหยุด × สาขา และ เสาร์-อาทิตย์ × สาขา (Attribute construction)"""
    out = df.copy()
    weekend = out[WEEKEND_COLS].sum(axis=1)
    for b in BRANCHES:
        out[f"weekend_{b}"] = weekend * out[f"branch_{b}"]
        out[f"holiday_{b}"] = out["is_holiday"] * out[f"branch_{b}"]
    return out


def fit_scaler(train: pd.DataFrame) -> dict:
    """ค่าเฉลี่ยและ SD สำหรับ Z-score คำนวณจากข้อมูลฝึกเท่านั้น (กัน Data leakage)"""
    return {c: (float(train[c].mean()), float(train[c].std(ddof=0))) for c in SCALED}


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (mean, sd) in scaler.items():
        out[f"{c}_z"] = (out[c] - mean) / sd
    return out


# ---------------------------------------------------------------------------
# โมเดลและ Baseline
# ---------------------------------------------------------------------------
def fit_model(train: pd.DataFrame, name: str) -> LinearRegression:
    """
    fit_intercept=False: One-hot สาขา 3 คอลัมน์รวมกันได้ 1 เสมอ ถ้ามีค่าคงที่ด้วยจะเกิด Dummy variable trap
    3 คอลัมน์สาขาจึงทำหน้าที่เป็น "ยอดตั้งต้นของแต่ละสาขา" แทนค่าคงที่
    """
    return LinearRegression(fit_intercept=False).fit(train[FEATURES[name]], train["revenue"])


def predict_all(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """ทำนายด้วย Baseline 2 แบบ และโมเดล 2 แบบ (Z-score จากข้อมูลฝึก)"""
    scaler = fit_scaler(train)
    tr, te = apply_scaler(train, scaler), apply_scaler(test, scaler)
    dow_mean = tr.groupby(["branch", "day_of_week"])["revenue"].mean()
    out = te[["date", "branch", "revenue"]].copy()
    out["B1_last_week"] = te["lag7_revenue"]
    out["B2_branch_dow_mean"] = [dow_mean.get((b, d), np.nan) for b, d in zip(te["branch"], te["day_of_week"])]
    for name in FEATURES:
        out[name] = fit_model(tr, name).predict(te[FEATURES[name]])
    return out


METHOD_TH = {"B1_last_week": "Baseline: เท่ากับสัปดาห์ก่อน", "B2_branch_dow_mean": "Baseline: ค่าเฉลี่ยสาขา × วันในสัปดาห์",
             "M1": "Linear Regression พื้นฐาน", "M2": "Linear Regression + Interaction"}


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    # ถ้ามีวันที่ทายไม่ได้ (ค่าว่าง) จำนวนวันที่เทียบจะไม่เท่ากันระหว่างวิธี ต้องแจ้ง ไม่ข้ามเงียบ ๆ
    if pred.isna().any() or actual.isna().any():
        raise ValueError("มีค่าว่างในยอดจริงหรือยอดที่ทาย เทียบวิธีต่าง ๆ อย่างยุติธรรมไม่ได้")
    err = pred - actual
    mae = float(err.abs().mean())
    ss_res, ss_tot = float((err ** 2).sum()), float(((actual - actual.mean()) ** 2).sum())
    return {"MAE": mae, "MAE_pct_of_mean": mae / float(actual.mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": 1 - ss_res / ss_tot, "bias": float(err.mean())}


def score_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = [{"method": m, "method_th": METHOD_TH[m], **metrics(pred["revenue"], pred[m])} for m in METHOD_TH]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ขั้นตอนหลัก: เลือกโมเดล -> ทดสอบ -> Walk-forward
# ---------------------------------------------------------------------------
def select_model(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """ฝึก ม.ค.-เม.ย. แล้วเลือกโมเดล (M1 / M2) ที่ MAE ต่ำสุดในเดือน พ.ค. (ไม่ดูข้อมูล มิ.ย.)"""
    train = df[df["date"].dt.month.isin(TRAIN_MONTHS)]
    val = df[df["date"].dt.month == VALIDATION_MONTH]
    scores = score_table(predict_all(train, val))
    models = scores[scores["method"].isin(FEATURES)]
    return models.sort_values("MAE").iloc[0]["method"], scores


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    ทดสอบซ้ำหลายเดือนแบบ Nested (กัน Data leakage):
    ทดสอบเดือน m -> เลือกโมเดล (M1 / M2) ด้วยเดือน m-1 โดยฝึกจากเดือนก่อนหน้านั้น
                 -> ฝึกโมเดลที่เลือกใหม่ด้วยทุกเดือนก่อน m -> ทดสอบเดือน m
    ทุกการตัดสินใจใช้เฉพาะข้อมูลก่อนเดือนที่ทดสอบ
    """
    month = df["date"].dt.month
    rows = []
    for m in [4, 5, 6]:
        val = score_table(predict_all(df[month < m - 1], df[month == m - 1]))
        chosen = val[val["method"].isin(FEATURES)].sort_values("MAE").iloc[0]["method"]
        s = score_table(predict_all(df[month < m], df[month == m])).set_index("method")["MAE"]
        rows.append({"test_month": m, "chosen_model": chosen, "MAE_B1": s["B1_last_week"],
                     "MAE_B2": s["B2_branch_dow_mean"], "MAE_model": s[chosen],
                     "beats_both_baselines": bool(s[chosen] < min(s["B1_last_week"], s["B2_branch_dow_mean"]))})
    return pd.DataFrame(rows)


def coefficients(train: pd.DataFrame, name: str) -> pd.DataFrame:
    """ตาราง coefficient อ่านเป็น insight ได้ (หน่วยบาทต่อวัน)"""
    scaler = fit_scaler(train)
    model = fit_model(apply_scaler(train, scaler), name)
    out = pd.DataFrame({"feature": FEATURES[name], "coefficient_baht": model.coef_})
    out["meaning_th"] = out["feature"].map(FEATURE_TH)
    # แปลงตัวแปร Z-score กลับเป็นหน่วยเดิม เพื่อให้อ่านง่าย
    per_unit = {"lag7_revenue_z": ("ต่อยอดสัปดาห์ก่อน 1 บาท", scaler["lag7_revenue"][1]),
                "trend_z": ("ต่อ 1 วันที่ผ่านไป", scaler["trend"][1])}
    out["per_original_unit"] = [out.loc[i, "coefficient_baht"] / per_unit[f][1] if f in per_unit else np.nan
                                for i, f in enumerate(out["feature"])]
    out["original_unit"] = out["feature"].map(lambda f: per_unit[f][0] if f in per_unit else "")
    return out


def residual_analysis(pred: pd.DataFrame, chosen: str) -> dict:
    """ความคลาดเคลื่อนของโมเดลที่เลือก: แยกสาขา / สัปดาห์แรกของเดือน / วันที่พลาดมากที่สุด"""
    p = pred.assign(error=pred[chosen] - pred["revenue"], abs_error=(pred[chosen] - pred["revenue"]).abs())
    by_branch = p.groupby("branch").agg(MAE=("abs_error", "mean"), bias=("error", "mean"),
                                        mean_revenue=("revenue", "mean")).reset_index()
    p["week_of_month"] = (p["date"].dt.day - 1) // 7 + 1
    by_week = p.pivot_table(index="week_of_month", columns="branch", values="error", aggfunc="mean").reset_index()
    worst = p.sort_values("abs_error", ascending=False).head(5)[["date", "branch", "revenue", chosen, "error"]]
    return {"by_branch": by_branch, "bias_by_week": by_week, "worst_days": worst, "errors": p}


def forecast_next_week(df: pd.DataFrame, chosen: str, db_path: Path = load.DB_PATH) -> pd.DataFrame:
    """
    ทำนาย 7 วันถัดจากข้อมูลล่าสุด (1-7 ก.ค. 2569) ด้วยโมเดลที่ฝึกจากข้อมูลทั้งหมด
    ใช้ยอดของ 7 วันก่อนจาก agg_daily / สมมุติว่าไม่มีวันหยุด (ไฟล์วันหยุดมีถึง มิ.ย.)
    """
    daily = load.query("SELECT branch, date, revenue FROM agg_daily", db_path)
    daily["date"] = pd.to_datetime(daily["date"])
    start = daily["date"].max() + pd.Timedelta(days=1)
    origin = daily["date"].min()   # จุดเริ่มของตัวแปร trend (ตรงกับโมดูล 5)
    rows = []
    for d in pd.date_range(start, periods=FORECAST_DAYS):
        for b in BRANCHES:
            lag = daily[(daily["branch"] == b) & (daily["date"] == d - pd.Timedelta(days=7))]["revenue"]
            row = {"date": d, "branch": b, "lag7_revenue": float(lag.iloc[0]) if len(lag) else np.nan,
                   "trend": (d - origin).days, "is_holiday": 0, "day_of_week": d.dayofweek}
            for c in BRANCH_COLS:
                row[c] = int(c == f"branch_{b}")
            for c in WEEKDAY_COLS + WEEKEND_COLS:
                row[c] = int(c == f"dow_{d.day_name()[:3]}")
            rows.append(row)
    fut = add_interactions(pd.DataFrame(rows))
    scaler = fit_scaler(df)
    model = fit_model(apply_scaler(df, scaler), chosen)
    fut = apply_scaler(fut, scaler)
    ok = fut["lag7_revenue"].notna()
    fut["predicted_revenue"] = np.nan
    fut.loc[ok, "predicted_revenue"] = model.predict(fut.loc[ok, FEATURES[chosen]])
    return fut[["date", "branch", "lag7_revenue", "predicted_revenue"]]


# ---------------------------------------------------------------------------
# บันทึกลง SQLite (ตารางผลลัพธ์ของโมเดล อยู่ใน load.DERIVED_TABLES)
# ---------------------------------------------------------------------------
def save_tables(tables: dict[str, tuple[pd.DataFrame, list[str]]], db_path: Path = load.DB_PATH) -> None:
    with closing(load.connect(db_path)) as con:
        with con:   # Transaction เดียว
            for name, (df, pk) in tables.items():
                con.execute(f"DROP TABLE IF EXISTS {name}")
                con.execute(load.ddl_from_frame(name, df, pk))
                load._insert(con, name, df)


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
def run_forecast(db_path: Path = load.DB_PATH, save: bool = True) -> dict:
    df = load_features(db_path)
    chosen, validation = select_model(df)

    # ฝึกใหม่ด้วย ม.ค.-พ.ค. แล้วทดสอบ มิ.ย. ครั้งเดียว
    train = df[df["date"].dt.month < TEST_MONTH]
    test = df[df["date"].dt.month == TEST_MONTH]
    pred = predict_all(train, test)
    result = {
        "chosen": chosen, "validation": validation, "test": score_table(pred), "predictions": pred,
        "walk_forward": walk_forward(df), "coefficients": coefficients(train, chosen),
        "residuals": residual_analysis(pred, chosen), "next_week": forecast_next_week(df, chosen, db_path),
        "n_train": len(train), "n_test": len(test),
    }
    if save:
        as_date = lambda d: d.assign(date=d["date"].dt.strftime("%Y-%m-%d"))
        daily = as_date(pred.rename(columns={"revenue": "actual_revenue"}).assign(chosen_model=chosen))
        save_tables({
            "forecast_daily": (daily, ["branch", "date"]),
            "forecast_coefficients": (result["coefficients"], ["feature"]),
            "forecast_next_week": (as_date(result["next_week"]), ["branch", "date"]),
        }, db_path)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 45)
    res = run_forecast()
    fmt = lambda t: t.round({"MAE": 1, "MAE_pct_of_mean": 3, "RMSE": 1, "R2": 3, "bias": 1})
    print("=== เลือกโมเดลด้วยเดือน พ.ค. (Validation) ===")
    print(fmt(res["validation"]).to_string(index=False))
    print(f"-> เลือก {res['chosen']}")
    print(f"\n=== ทดสอบกับ มิ.ย. (ฝึก {res['n_train']} / ทดสอบ {res['n_test']} สาขา-วัน) ===")
    print(fmt(res["test"]).to_string(index=False))
    print("\n=== Walk-forward ===")
    print(res["walk_forward"].round(1).to_string(index=False))
    print("\n=== Coefficient ===")
    print(res["coefficients"].round(3).to_string(index=False))
    r = res["residuals"]
    print("\n=== ความคลาดเคลื่อนแยกสาขา (มิ.ย.) ===")
    print(r["by_branch"].round(1).to_string(index=False))
    print("\n=== ความคลาดเคลื่อนเฉลี่ยรายสัปดาห์ของ มิ.ย. (บวก = ทายสูงเกิน / ลบ = ทายต่ำเกิน) ===")
    print(r["bias_by_week"].round(0).to_string(index=False))
    print("\n=== 5 วันที่พลาดมากที่สุด ===")
    print(r["worst_days"].round(0).to_string(index=False))
    print("\n=== ทำนาย 7 วันถัดไป ===")
    print(res["next_week"].round(0).to_string(index=False))
