"""
โมดูล 5: แปลงข้อมูลให้พร้อมใช้ (Transform)

โมดูล 4 ทำให้ข้อมูล "ถูกต้อง" แล้ว โมดูลนี้ทำให้ข้อมูล "พร้อมใช้งาน" สำหรับงานปลายทาง
    โมดูล 6 Load           -> Star schema (fact + dimension) พร้อมกุญแจเชื่อมตาราง
    โมดูล 7 Recommendation -> ตาราง basket (บิล × เมนู)
    โมดูล 8 Regression     -> model_features (encode + normalize แล้ว)
    โมดูล 9 Dashboard      -> ยอดสรุปรายวัน / รายชั่วโมง / รายเมนู

เทคนิค (Ch2 Data Transformation ทั้งหมด ยกเว้น T8)
    T1 Enrichment               T5 Aggregation
    T2 Attribute construction   T6 Encoding (One-hot / Dummy / Ordinal)
    T3 Discretization           T7 Normalization (Min-Max / Z-score)
    T4 Anonymization            T8 Data modeling: Star schema (Ch1, Ch5 Data warehouse)

วิธีรัน:  py -3.13 -m pipeline.transform
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import clean, extract, quality

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "transform"

# T4: Salt สำหรับ hash เบอร์โทร
# โปรเจกต์นี้เก็บไว้ในโค้ดเพื่อให้รันซ้ำได้ งานจริงต้องเก็บเป็นความลับแยกจากโค้ด (เช่น environment variable)
PHONE_SALT = "doi-brew-2569-demo-salt"

# T3: ช่วงของการแบ่งกลุ่ม
AGE_BINS = [0, 22, 35, 50, 150]                       # (0,22] (22,35] (35,50] (50,150]
AGE_LABELS = ["<=22", "23-35", "36-50", "51+"]
TIME_OF_DAY = {"เช้า": range(7, 11), "กลางวัน": range(11, 14), "บ่าย": range(14, 17), "เย็น": range(17, 20)}
SPEND_LABELS = ["ต่ำ", "กลาง", "สูง"]

# T6: Ordinal encoding (ข้อมูลที่มีลำดับจริง)
SIZE_ORDINAL = {"S": 1, "M": 2, "L": 3}
AGE_GROUP_ORDINAL = {label: i + 1 for i, label in enumerate(AGE_LABELS)}

BRANCH_INFO = [  # dim_branch
    ("nimman", "นิมมาน", "Nimman", "CSV"),
    ("cmu", "มช.", "CMU", "Excel"),
    ("thaphae", "ท่าแพ", "Tha Phae", "JSON"),
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# T5: กฎแยก "ข้อมูลหาย" ออกจาก "ไม่มีลูกค้าจริง" สำหรับวันที่ยอดขายเป็น 0
MISSING_WINDOW_DAYS = 14     # ดูวันประเภทเดียวกันในช่วง ±14 วัน (2 สัปดาห์)
MISSING_ALPHA = 0.01         # ระดับนัยสำคัญ 1%
# ตรวจหลายวันพร้อมกัน (สาขา × วัน) ถ้าใช้ 1% ตรง ๆ จะเจอวันที่ "ดูผิดปกติ" โดยบังเอิญหลายวัน
# จึงหารด้วยจำนวนวันที่ตรวจ (Bonferroni correction) ในฟังก์ชัน mark_missing_days


# ---------------------------------------------------------------------------
# T2 Attribute construction + T3 Discretization (ระดับรายการขาย)
# ---------------------------------------------------------------------------
def time_of_day(hour: pd.Series) -> pd.Series:
    """T3: ชั่วโมง -> ช่วงเวลาของวัน (นอกเวลาเปิดร้านได้ค่าว่าง)"""
    out = pd.Series(pd.NA, index=hour.index, dtype="object")
    for label, hours in TIME_OF_DAY.items():
        out[hour.isin(list(hours))] = label
    return out


def add_sale_attributes(sales: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """T2 สร้างคอลัมน์จากวันเวลา + T1 Enrichment วันหยุด + T3 ช่วงเวลาของวัน"""
    s = sales.copy()
    dt = s["sale_datetime"]
    s["date"] = dt.dt.date
    s["date_key"] = dt.dt.strftime("%Y%m%d").astype(int)
    s["month"] = dt.dt.month
    s["day_of_week"] = dt.dt.dayofweek
    s["hour"] = dt.dt.hour
    # เวลาของแถวที่ติดธง time_suspect เชื่อไม่ได้ จึงไม่ใส่ช่วงเวลา
    s["time_of_day"] = time_of_day(s["hour"]).where(~s["time_suspect"], pd.NA)
    s["items_in_bill"] = s.groupby("receipt_id")["receipt_id"].transform("size")
    s["size_ordinal"] = s["size"].map(SIZE_ORDINAL).astype("Int64")   # T6 Ordinal
    hol = set(holidays["date"].dt.date)
    s["is_holiday"] = s["date"].isin(hol)
    return s


# ---------------------------------------------------------------------------
# ตาราง Dimension
# ---------------------------------------------------------------------------
def build_dim_date(holidays: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(quality.PERIOD_START.date(), quality.PERIOD_END.date(), freq="D")
    names = holidays.set_index(holidays["date"].dt.date)["name_th"]
    d = pd.DataFrame({"date": days.date})
    d["date_key"] = days.strftime("%Y%m%d").astype(int)
    d["year"], d["month"], d["day"] = days.year, days.month, days.day
    d["day_of_week"] = days.dayofweek
    d["day_name"] = [DAY_NAMES[i] for i in days.dayofweek]
    d["is_weekend"] = days.dayofweek >= 5
    d["holiday_name"] = d["date"].map(names)                  # T1 Enrichment
    d["is_holiday"] = d["holiday_name"].notna()
    return d[["date_key", "date", "year", "month", "day", "day_of_week", "day_name",
              "is_weekend", "is_holiday", "holiday_name"]]


def build_dim_branch() -> pd.DataFrame:
    return pd.DataFrame(BRANCH_INFO, columns=["branch_id", "name_th", "name_en", "source_format"])


def build_dim_product(products: pd.DataFrame) -> pd.DataFrame:
    return products[["product_id", "name_th", "name_en", "category",
                     "price_S", "price_M", "price_L", "price_nosize"]].copy()


def hash_phone(phone: pd.Series) -> pd.Series:
    """T4: เข้ารหัสทางเดียว SHA-256 + Salt (ถอดกลับเป็นเบอร์จริงไม่ได้ แต่เบอร์เดียวกันได้ค่าเดียวกันเสมอ)"""
    return phone.map(lambda p: hashlib.sha256((PHONE_SALT + str(p)).encode("utf-8")).hexdigest())


def build_dim_member(members: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    ตารางสมาชิกสำหรับวิเคราะห์
      T2 อายุจากปีเกิด / T3 กลุ่มอายุ + ระดับการใช้จ่าย / T6 Ordinal กลุ่มอายุ
      T1 Enrichment: ยอดซื้อสะสมของลูกค้า (ตัวอย่างใน Ch2 "rolled up into a grand total")
      T4 Anonymization: ลบชื่อ + hash เบอร์โทร
    """
    m = members.copy()
    m["age"] = (clean.REFERENCE_YEAR - m["birth_year"]).astype("Int64")
    m["age_group"] = pd.cut(m["age"].astype(float), bins=AGE_BINS, labels=AGE_LABELS).astype(object)
    m["age_group"] = m["age_group"].where(m["age_group"].notna(), pd.NA)   # อายุว่าง -> กลุ่มว่าง (ไม่ใช่ "nan")
    m["age_group_ordinal"] = m["age_group"].map(AGE_GROUP_ORDINAL).astype("Int64")

    # T1: รวมยอดซื้อของสมาชิกแต่ละคน
    s = sales[sales["member_id"].notna()].copy()
    s["category"] = s["product_id"].map(products.set_index("product_id")["category"])
    g = s.groupby("member_id")
    roll = pd.DataFrame({
        "total_spent": g["line_total"].sum(),
        "n_bills": g["receipt_id"].nunique(),
        "n_items": g["qty"].sum(),
        "first_purchase": g["sale_datetime"].min().dt.date,
        "last_purchase": g["sale_datetime"].max().dt.date,
        "favorite_category": s.groupby(["member_id", "category"])["qty"].sum()
                              .groupby(level=0).idxmax().map(lambda k: k[1]),
        "favorite_branch": s.groupby(["member_id", "branch"])["receipt_id"].nunique()
                            .groupby(level=0).idxmax().map(lambda k: k[1]),
    })
    m = m.merge(roll, left_on="member_id", right_index=True, how="left")
    m["total_spent"] = m["total_spent"].fillna(0.0)
    for c in ["n_bills", "n_items"]:
        m[c] = m[c].fillna(0).astype(int)

    # T3: ระดับการใช้จ่ายแบบ Equal-frequency (แต่ละกลุ่มมีคนเท่า ๆ กัน) เพราะยอดใช้จ่ายเบ้ขวา
    #     ถ้าแบ่งช่วงกว้างเท่ากัน (Equal-width) กลุ่ม "สูง" จะมีคนน้อยมากจนเปรียบเทียบไม่ได้
    buyers = m["n_bills"] > 0
    ranks = m.loc[buyers, "total_spent"].rank(method="first")
    m["spend_tier"] = "ไม่เคยซื้อ"
    if buyers.sum() >= len(SPEND_LABELS):   # ผู้ซื้อน้อยกว่าจำนวนกลุ่ม แบ่งกลุ่มไม่ได้
        m.loc[buyers, "spend_tier"] = pd.qcut(ranks, q=len(SPEND_LABELS), labels=SPEND_LABELS).astype(str)

    # T4: ลบชื่อ + hash เบอร์โทร
    m["phone_hash"] = hash_phone(m["phone"])
    m = m.drop(columns=["full_name", "phone"])
    cols = ["member_id", "phone_hash", "gender", "birth_year", "birth_year_imputed", "age", "age_group",
            "age_group_ordinal", "home_branch", "register_date", "total_spent", "n_bills", "n_items",
            "spend_tier", "first_purchase", "last_purchase", "favorite_category", "favorite_branch"]
    return m[cols]


# ---------------------------------------------------------------------------
# ตาราง Fact
# ---------------------------------------------------------------------------
def build_fact_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """1 แถว = 1 รายการสินค้า เก็บเฉพาะกุญแจไปยังตาราง dimension + ตัวเลข (measures)"""
    f = sales.reset_index(drop=True)
    f.insert(0, "sale_line_id", np.arange(1, len(f) + 1))
    f = f.rename(columns={"branch": "branch_id"})
    return f[["sale_line_id", "receipt_id", "date_key", "sale_datetime", "hour", "time_of_day",
              "branch_id", "product_id", "member_id", "size", "size_ordinal", "qty", "unit_price",
              "line_total", "items_in_bill", "time_suspect", "clean_flags"]]


# ---------------------------------------------------------------------------
# T5 Aggregation
# ---------------------------------------------------------------------------
def mark_missing_days(daily: pd.DataFrame) -> pd.DataFrame:
    """
    วันที่ยอดขายเป็น 0: แยกว่า "ข้อมูลหาย" หรือ "ไม่มีลูกค้าจริง"
    ดูวันประเภทเดียวกัน (วันทำงาน / วันหยุด-เสาร์อาทิตย์) ของสาขาเดียวกันในช่วง ±21 วัน
    ถ้าปกติขายได้เฉลี่ย λ บิล โอกาสขายได้ 0 บิล = e^(-λ) (การแจกแจงแบบปัวซง)
    โอกาสต่ำกว่าเกณฑ์ -> ข้อมูลหาย (ไม่ใช่ 0 จริง) / ไม่ต่ำ -> ไม่มีลูกค้าจริง
    เกณฑ์ = 1% ÷ จำนวนสาขา-วันที่ตรวจ (Bonferroni) เพราะตรวจหลายวันพร้อมกัน
    """
    d = daily.copy()
    threshold = MISSING_ALPHA / len(d)
    d["day_type"] = np.where(d["is_weekend"] | d["is_holiday"], "off", "work")
    d["expected_bills"] = np.nan
    d["p_zero"] = np.nan
    zero = d["bills"] == 0
    for i in d.index[zero]:
        r = d.loc[i]
        near = d[(d["branch"] == r["branch"]) & (d["day_type"] == r["day_type"]) & (d["bills"] > 0)
                 & ((pd.to_datetime(d["date"]) - pd.Timestamp(r["date"])).abs().dt.days <= MISSING_WINDOW_DAYS)]
        lam = near["bills"].mean() if len(near) else 0.0
        d.loc[i, "expected_bills"] = lam
        d.loc[i, "p_zero"] = float(np.exp(-lam))
    d["data_missing"] = zero & (d["p_zero"] < threshold)
    for c in ["bills", "items", "revenue", "avg_bill"]:
        d.loc[d["data_missing"], c] = np.nan
    return d.drop(columns="day_type")


def min_max(x: pd.Series) -> pd.Series:
    """T7 Min-Max ไปอยู่ช่วง 0-1 (ถ้าทุกค่าเท่ากันจะได้ 0 แทนการหารด้วย 0)"""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else x * 0.0


def build_agg_daily(sales: pd.DataFrame, dim_date: pd.DataFrame) -> pd.DataFrame:
    """ยอดขายรายวันต่อสาขา (ครบทุกวันทุกสาขา วันที่ไม่มีขายได้ 0 หรือค่าว่างถ้าข้อมูลหาย)"""
    grid = pd.MultiIndex.from_product([list(extract.FILES), dim_date["date"]], names=["branch", "date"])
    g = sales.groupby(["branch", "date"])
    agg = pd.DataFrame({"bills": g["receipt_id"].nunique(), "items": g["qty"].sum(),
                        "revenue": g["line_total"].sum()}).reindex(grid, fill_value=0).reset_index()
    agg["avg_bill"] = (agg["revenue"] / agg["bills"]).where(agg["bills"] > 0, 0.0)
    agg = agg.merge(dim_date[["date", "date_key", "month", "day_of_week", "day_name", "is_weekend",
                              "is_holiday"]], on="date")
    agg = mark_missing_days(agg)
    # T7 Min-Max รายสาขา: เทียบ "รูปแบบ" ยอดขายของสาขาที่ขนาดต่างกันในกราฟเดียวได้
    agg["revenue_minmax"] = agg.groupby("branch")["revenue"].transform(min_max)
    return agg


def build_agg_hourly(sales: pd.DataFrame) -> pd.DataFrame:
    """ยอดขายรายชั่วโมงต่อสาขา (ไม่นับแถวที่เวลาเชื่อไม่ได้ time_suspect)"""
    s = sales[~sales["time_suspect"]]
    hours = range(quality.OPEN_HOUR, quality.CLOSE_HOUR)
    grid = pd.MultiIndex.from_product([list(extract.FILES), hours], names=["branch", "hour"])
    g = s.groupby(["branch", "hour"])
    out = pd.DataFrame({"bills": g["receipt_id"].nunique(), "revenue": g["line_total"].sum()}) \
        .reindex(grid, fill_value=0).reset_index()
    out["time_of_day"] = time_of_day(out["hour"])
    return out


def build_agg_product(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """ยอดขายรายเมนู × สาขา × เดือน"""
    months = range(quality.PERIOD_START.month, quality.PERIOD_END.month + 1)
    grid = pd.MultiIndex.from_product([products["product_id"], list(extract.FILES), months],
                                      names=["product_id", "branch", "month"])
    g = sales.groupby(["product_id", "branch", "month"])
    out = pd.DataFrame({"qty": g["qty"].sum(), "revenue": g["line_total"].sum()}) \
        .reindex(grid, fill_value=0).reset_index()
    return out.merge(products[["product_id", "name_th", "category"]], on="product_id")


def build_basket(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    ตาราง บิล × เมนู (มี = 1 / ไม่มี = 0) สำหรับ Recommendation
    แนวคิดเดียวกับ One-hot / Bag of Words ใน Ch7 แต่ "คำ" คือเมนูในบิล
    """
    b = pd.crosstab(sales["receipt_id"], sales["product_id"]).clip(upper=1)
    b = b.reindex(columns=products["product_id"], fill_value=0).astype("int8")
    branch = sales.drop_duplicates("receipt_id").set_index("receipt_id")["branch"]
    b.insert(0, "branch", branch.reindex(b.index))
    return b.reset_index()


# ---------------------------------------------------------------------------
# T6 Encoding + T7 Normalization สำหรับ Regression
# ---------------------------------------------------------------------------
def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


def build_model_features(agg_daily: pd.DataFrame) -> pd.DataFrame:
    """
    ตัวแปรสำหรับทำนายยอดขายรายวัน (โมดูล 8)
      T6 One-hot สาขา (nominal) / Dummy วันในสัปดาห์ (ตัดวันจันทร์ กัน Dummy variable trap)
         หมายเหตุสำหรับโมดูล 8: One-hot สาขาครบ 3 คอลัมน์รวมกันได้ 1 เสมอ ถ้าโมเดลมี intercept จะเกิด
         Dummy variable trap เช่นกัน จึงต้องใช้ fit_intercept=False ให้ 3 คอลัมน์นี้เป็นค่าตั้งต้นของแต่ละสาขา
      T7 Z-score ตัวเลข เพื่อให้เทียบขนาด coefficient ได้
    ไม่ใช้วันที่ข้อมูลหาย และไม่ใส่ is_weekend เพราะรู้ได้จาก dummy วันในสัปดาห์อยู่แล้ว (ซ้ำซ้อน)
    """
    d = agg_daily.sort_values(["branch", "date"]).copy()
    # ยอดขายวันเดียวกันของสัปดาห์ก่อน (ค่าในอดีตที่รู้แล้วตอนทำนาย)
    d["lag7_revenue"] = d.groupby("branch")["revenue"].shift(7)
    d["trend"] = (pd.to_datetime(d["date"]) - pd.to_datetime(d["date"]).min()).dt.days
    d = d[~d["data_missing"] & d["lag7_revenue"].notna()].copy()

    onehot = pd.get_dummies(d["branch"], prefix="branch").astype(int)                       # One-hot
    dummy = pd.get_dummies(pd.Categorical(d["day_name"], categories=DAY_NAMES), prefix="dow",
                           drop_first=True).astype(int).set_axis(d.index)                     # Dummy
    out = pd.concat([d[["date", "date_key", "branch", "revenue", "bills", "month", "is_holiday",
                        "lag7_revenue", "trend"]], onehot, dummy], axis=1)
    out["is_holiday"] = out["is_holiday"].astype(int)
    out["lag7_revenue_z"] = zscore(out["lag7_revenue"])                                       # Z-score
    out["trend_z"] = zscore(out["trend"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
TABLES = ["fact_sales", "dim_date", "dim_product", "dim_branch", "dim_member",
          "agg_daily", "agg_hourly", "agg_product", "basket", "model_features"]


def run_transform(save: bool = True, cleaned: dict | None = None) -> dict:
    cleaned = cleaned or clean.run_clean(save=False)
    products, holidays = cleaned["products"], cleaned["holidays"]
    sales = add_sale_attributes(cleaned["sales"], holidays)

    out = {
        "dim_date": build_dim_date(holidays),
        "dim_branch": build_dim_branch(),
        "dim_product": build_dim_product(products),
        "dim_member": build_dim_member(cleaned["members"], sales, products),
        "fact_sales": build_fact_sales(sales),
        "agg_hourly": build_agg_hourly(sales),
        "agg_product": build_agg_product(sales, products),
        "basket": build_basket(sales, products),
    }
    out["agg_daily"] = build_agg_daily(sales, out["dim_date"])
    out["model_features"] = build_model_features(out["agg_daily"])
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            out[name].to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    res = run_transform()
    pd.set_option("display.width", 200)
    print("=== ตารางที่ได้ ===")
    for name in TABLES:
        print(f"  {name:15s} {len(res[name]):>6,} แถว x {res[name].shape[1]:>2} คอลัมน์")
    d = res["agg_daily"]
    print("\n=== วันที่ยอดขายเป็น 0 ===")
    print(d[d["expected_bills"].notna()][["branch", "date", "day_name", "is_holiday", "expected_bills", "p_zero",
                                          "data_missing"]].to_string(index=False))
    m = res["dim_member"]
    print("\n=== กลุ่มอายุ / ระดับการใช้จ่าย ===")
    print(m["age_group"].value_counts().sort_index().to_dict(), m["spend_tier"].value_counts().to_dict())
    print(f"\nยอดขายรวม fact = {res['fact_sales']['line_total'].sum():,.0f} / agg_daily = "
          f"{d['revenue'].sum():,.0f} / agg_product = {res['agg_product']['revenue'].sum():,.0f}")
