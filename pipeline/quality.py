"""
โมดูล 3: วัดคุณภาพข้อมูล (Data Quality Assessment)

"ตรวจสุขภาพก่อนรักษา" วัดคุณภาพของตาราง staging (จากโมดูล 2) เป็นตัวเลข 6 มิติตาม Ch3
โมดูลนี้ "วัดอย่างเดียว ไม่แก้ข้อมูล" การแก้อยู่ในโมดูล 4 ซึ่งจะเรียก assess() ซ้ำกับข้อมูลที่สะอาดแล้ว
เพื่อแสดงคะแนนก่อน / หลัง (Ch3 ขั้น Control, Ch4 ขั้น Validation)

กระบวนการ 6 ขั้นของ Ch3 ที่ทำในโมดูลนี้
    ขั้น 1 Definition  -> RULES (กฎคุณภาพ) + TARGET (เป้าหมายคะแนน)
    ขั้น 2 Assessment  -> assess() รันทุกกฎแล้วคิดคะแนนด้วย Simple ratio
    ขั้น 3 Analysis    -> by_branch() แยกผลตามสาขาเพื่อหาสาเหตุ

สูตรคะแนน (Ch3 Data Quality Assessment)
    ระดับกฎ   : Simple ratio = 1 - (จำนวนที่ผิด / จำนวนที่ตรวจ)
    ระดับมิติ  : Weighted average (น้ำหนักเท่ากัน) และ Min operation (ค่าต่ำสุด)
    Timeliness : max(0, 1 - Currency / Volatility)

วิธีรัน:  py -3.13 -m pipeline.quality
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from pipeline import extract

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ---------------------------------------------------------------------------
# ขั้น 1 Definition: กฎธุรกิจและเป้าหมาย
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2026-01-01", tz=extract.TZ)
PERIOD_END = pd.Timestamp("2026-06-30", tz=extract.TZ)
OPEN_HOUR, CLOSE_HOUR = 7, 20          # ร้านเปิด 07:00-20:00
AGE_MIN, AGE_MAX = 10, 100             # อายุสมาชิกที่เป็นไปได้
VOLATILITY_DAYS = 7                    # ข้อมูลยอดขายใช้ได้ 7 วัน (เจ้าของร้านดูรายงานรายสัปดาห์)
DEFAULT_RECEIVED_AT = pd.Timestamp("2026-07-01 07:05", tz=extract.TZ)  # วันที่รับข้อมูล (ถ้าไฟล์ไม่ระบุ)
TARGET = 0.98                          # เป้าหมายคะแนนของทุกมิติ
PHONE_STANDARD = r"^0\d{9}$"           # รูปแบบเบอร์โทรมาตรฐาน 0XXXXXXXXX
PRICE_TOLERANCE = 0.01                 # ราคาคลาดจากตารางไม่เกิน 1 สตางค์ถือว่าตรง (เผื่อทศนิยมจากการหาร)

DIMENSIONS = ["Completeness", "Uniqueness", "Validity", "Consistency", "Accuracy", "Timeliness"]
DIMENSION_TH = {
    "Completeness": "ความครบถ้วน", "Uniqueness": "ความไม่ซ้ำ", "Validity": "ความถูกต้องตามกฎ",
    "Consistency": "ความสอดคล้อง", "Accuracy": "ความแม่นยำ", "Timeliness": "ความทันเวลา",
}


@dataclass
class Context:
    """ข้อมูลทั้งหมดที่กฎต้องใช้"""
    sales: pd.DataFrame
    members: pd.DataFrame
    products: pd.DataFrame
    received_at: pd.Timestamp


@dataclass
class Rule:
    """
    กฎคุณภาพ 1 ข้อ
    check() คืน Series ค่าจริง/เท็จ 1 ค่าต่อ 1 สิ่งที่ตรวจ (True = ผิดกฎ)
    index ของ Series คือแถวที่ตรวจ จำนวนที่ตรวจจึงเท่ากับความยาวของ Series
    กฎ Timeliness คืนคะแนนตรง ๆ (kind = "score") เพราะใช้สูตรคนละแบบ
    """
    rule_id: str
    dimension: str
    table: str
    description: str
    check: Callable[[Context], pd.Series]
    kind: str = "ratio"


# ---------------------------------------------------------------------------
# ฟังก์ชันช่วย
# ---------------------------------------------------------------------------
def _category(ctx: Context) -> pd.Series:
    """หมวดสินค้าของแต่ละแถวขาย (ว่างถ้าจับคู่สินค้าไม่ได้)"""
    return ctx.sales["product_id"].map(ctx.products.set_index("product_id")["category"])


def _normalize_name(s: pd.Series) -> pd.Series:
    """ตัดช่องว่างเกินเพื่อ 'เปรียบเทียบ' เท่านั้น ไม่ได้แก้ข้อมูลจริง"""
    return s.fillna("").str.split().str.join(" ")


def _normalize_phone(s: pd.Series) -> pd.Series:
    """เหลือแต่ตัวเลข และเปลี่ยน +66 เป็น 0 เพื่อ 'เปรียบเทียบ' เท่านั้น"""
    digits = s.fillna("").str.replace(r"\D", "", regex=True)
    return digits.str.replace(r"^66", "0", regex=True)


def expected_price(sales: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """ราคาตามตารางสินค้า (ค่าอ้างอิง) ของแต่ละแถว ตามสินค้าและขนาด"""
    p = products.set_index("product_id")
    out = pd.Series(float("nan"), index=sales.index)
    for size, col in [("S", "price_S"), ("M", "price_M"), ("L", "price_L")]:
        m = sales["size"] == size
        out[m] = sales.loc[m, "product_id"].map(p[col]).astype("float64")
    bakery = sales["product_id"].map(p["category"]) == "bakery"
    out[bakery] = sales.loc[bakery, "product_id"].map(p["price_nosize"]).astype("float64")
    return out


# ---------------------------------------------------------------------------
# กฎแต่ละข้อ
# ---------------------------------------------------------------------------
# ---- Completeness ----
def cp1_schema(ctx):
    cols = pd.Index(extract.STAGING_COLUMNS)
    return pd.Series(~cols.isin(ctx.sales.columns), index=cols)


def cp2_unit_price(ctx):
    return ctx.sales["unit_price"].isna()


def cp3_qty(ctx):
    return ctx.sales["qty"].isna()


def cp4_drink_size(ctx):
    # ตรวจเฉพาะเครื่องดื่ม เบเกอรี่ไม่มีขนาดอยู่แล้ว (ไม่ใช่ข้อมูลหาย)
    # แถวที่ไม่รู้ว่าเป็นสินค้าอะไร (CS1) ไม่ตรวจ เพราะไม่รู้ว่าควรมีขนาดหรือไม่
    drinks = _category(ctx).isin(["coffee", "non_coffee"])
    return ctx.sales.loc[drinks, "size"].isna()


def cp5_gender(ctx):
    return ctx.members["gender"].isna()


def cp6_birth_year(ctx):
    return ctx.members["birth_year"].isna()


def cp7_branch_days(ctx):
    # Population completeness: ทุกสาขาต้องมียอดขายทุกวัน (เหมือนตัวอย่าง 50 รัฐใน Ch3)
    days = pd.date_range(PERIOD_START.date(), PERIOD_END.date(), freq="D").date
    # ใช้รายชื่อสาขาที่ "ควรมี" (ไม่ใช่สาขาที่พบในข้อมูล) ถ้าสาขาใดหายทั้งสาขาจะได้จับได้
    expected = pd.MultiIndex.from_product([list(extract.FILES), days],
                                          names=["branch", "date"])
    present = pd.MultiIndex.from_frame(
        ctx.sales.assign(date=ctx.sales["sale_datetime"].dt.date)[["branch", "date"]].drop_duplicates())
    return pd.Series(~expected.isin(present), index=expected)


# ---- Uniqueness ----
def uq1_duplicate_rows(ctx):
    # แถวซ้ำทุกคอลัมน์ (ไม่นับตำแหน่งในไฟล์) แถวแรกถือเป็นตัวจริง แถวถัดไปคือตัวซ้ำ
    core = [c for c in ctx.sales.columns if c != "source_row"]
    return ctx.sales.duplicated(subset=core, keep="first")


def uq2_duplicate_bill_records(ctx):
    # ไฟล์ JSON ท่าแพ: 1 ระเบียน = 1 บิล เลขที่บิลเดียวกันต้องมีแค่ 1 ระเบียน
    tp = ctx.sales[ctx.sales["source_file"] == extract.FILES["thaphae"]]
    records = (tp.assign(record=tp["source_row"].str.extract(r"^(receipts\[\d+\])")[0])
                 .drop_duplicates("record")["receipt_id"])
    # index = เลขที่บิล เพื่อให้รายการที่ผิดแสดงเลขที่บิล (ไม่ใช่ตำแหน่งในไฟล์)
    return pd.Series(records.duplicated(keep="first").to_numpy(), index=records.to_numpy())


def uq3_member_id(ctx):
    return ctx.members["member_id"].duplicated(keep="first")


def uq4_member_name_phone(ctx):
    # กฎตัวอย่างใน Ch3: "Customer name and Address together should be unique"
    key = _normalize_name(ctx.members["full_name"]) + "|" + _normalize_phone(ctx.members["phone"])
    return key.duplicated(keep="first")


# ---- Validity ----
def vl1_qty_positive(ctx):
    qty = ctx.sales["qty"].dropna()
    return qty <= 0


def vl2_opening_hours(ctx):
    h = ctx.sales["sale_datetime"].dropna().dt.hour
    return (h < OPEN_HOUR) | (h >= CLOSE_HOUR)


def vl3_period(ctx):
    dt = ctx.sales["sale_datetime"].dropna()
    return (dt < PERIOD_START) | (dt >= PERIOD_END + pd.Timedelta(days=1))


def vl4_age(ctx):
    age = ctx.received_at.year - ctx.members["birth_year"].dropna().astype(int)
    return (age < AGE_MIN) | (age > AGE_MAX)


def vl5_gender_domain(ctx):
    return ~ctx.members["gender"].dropna().isin(["M", "F"])


# ---- Consistency ----
def cs1_product_in_master(ctx):
    s = ctx.sales
    checked = s["product_raw"].notna()
    return s.loc[checked, "product_id"].isna()


def cs2_member_in_members(ctx):
    mid = ctx.sales["member_id"].dropna()
    return ~mid.isin(ctx.members["member_id"])


def cs3_phone_format(ctx):
    return ~ctx.members["phone"].fillna("").str.match(PHONE_STANDARD)


# ---- Accuracy ----
def ac1_price_matches_list(ctx):
    # เทียบกับค่าอ้างอิง (ตารางราคา) ตรวจเฉพาะแถวที่รู้ทั้งราคาและราคาอ้างอิง
    exp = expected_price(ctx.sales, ctx.products)
    checked = ctx.sales["unit_price"].notna() & exp.notna()
    return (ctx.sales.loc[checked, "unit_price"] - exp[checked]).abs() > PRICE_TOLERANCE


# ---- Timeliness ----
def tm1_currency(ctx):
    # Timeliness = max(0, 1 - Currency / Volatility)  (Ch3)
    # Currency = จำนวนวันระหว่างวันที่ของข้อมูลล่าสุด กับวันที่รับข้อมูล
    latest = ctx.sales.groupby("branch")["sale_datetime"].max().dt.date
    currency = latest.map(lambda d: (ctx.received_at.date() - d).days)
    return (1 - currency / VOLATILITY_DAYS).clip(lower=0, upper=1)


RULES: list[Rule] = [
    Rule("CP1", "Completeness", "schema", "มีคอลัมน์ครบตามโครงสร้างกลาง (Schema completeness)", cp1_schema),
    Rule("CP2", "Completeness", "sales", "ราคาต่อชิ้นต้องไม่ว่าง", cp2_unit_price),
    Rule("CP3", "Completeness", "sales", "จำนวนต้องไม่ว่าง", cp3_qty),
    Rule("CP4", "Completeness", "sales", "เครื่องดื่มต้องมีขนาดแก้ว (เบเกอรี่ไม่ต้องมี)", cp4_drink_size),
    Rule("CP5", "Completeness", "members", "เพศต้องไม่ว่าง", cp5_gender),
    Rule("CP6", "Completeness", "members", "ปีเกิดต้องไม่ว่าง", cp6_birth_year),
    Rule("CP7", "Completeness", "branch-days", "ทุกสาขาต้องมียอดขายทุกวัน (Population completeness)", cp7_branch_days),
    Rule("UQ1", "Uniqueness", "sales", "ไม่มีแถวซ้ำทุกคอลัมน์", uq1_duplicate_rows),
    Rule("UQ2", "Uniqueness", "thaphae bills", "บิลเดียวกันต้องมีแค่ 1 ระเบียนในไฟล์ JSON", uq2_duplicate_bill_records),
    Rule("UQ3", "Uniqueness", "members", "รหัสสมาชิกต้องไม่ซ้ำ", uq3_member_id),
    Rule("UQ4", "Uniqueness", "members", "ชื่อ + เบอร์โทร ต้องไม่ซ้ำ (คนเดียวกันสมัครซ้ำ)", uq4_member_name_phone),
    Rule("VL1", "Validity", "sales", "จำนวนต้องมากกว่า 0", vl1_qty_positive),
    Rule("VL2", "Validity", "sales", f"เวลาขายต้องอยู่ในเวลาเปิดร้าน {OPEN_HOUR:02d}:00-{CLOSE_HOUR:02d}:00", vl2_opening_hours),
    Rule("VL3", "Validity", "sales", "วันที่ขายต้องอยู่ในช่วงข้อมูล 1 ม.ค.-30 มิ.ย. 2569", vl3_period),
    Rule("VL4", "Validity", "members", f"อายุต้องอยู่ระหว่าง {AGE_MIN}-{AGE_MAX} ปี", vl4_age),
    Rule("VL5", "Validity", "members", "เพศต้องเป็น M หรือ F", vl5_gender_domain),
    Rule("CS1", "Consistency", "sales", "สินค้าต้องมีอยู่ในตารางสินค้าหลัก", cs1_product_in_master),
    Rule("CS2", "Consistency", "sales", "รหัสสมาชิกในบิลต้องมีอยู่ในตารางสมาชิก", cs2_member_in_members),
    Rule("CS3", "Consistency", "members", "เบอร์โทรต้องเป็นรูปแบบเดียวกัน (0XXXXXXXXX)", cs3_phone_format),
    Rule("AC1", "Accuracy", "sales", "ราคาต่อชิ้นต้องตรงกับตารางราคา (ค่าอ้างอิง)", ac1_price_matches_list),
    Rule("TM1", "Timeliness", "branches", f"ข้อมูลต้องใหม่พอ (Volatility {VOLATILITY_DAYS} วัน)", tm1_currency, kind="score"),
]


# ---------------------------------------------------------------------------
# ขั้น 2 Assessment
# ---------------------------------------------------------------------------
@dataclass
class QualityResult:
    rules: pd.DataFrame        # 1 แถว = 1 กฎ
    dimensions: pd.DataFrame   # 1 แถว = 1 มิติ
    overall: dict              # คะแนนรวม + ตัวชี้วัดสรุป 4 ตัว
    details: pd.DataFrame      # รายการสิ่งที่ผิดกฎทุกรายการ (ส่งต่อให้โมดูล 4)
    by_branch: pd.DataFrame    # ขั้น 3 Analysis
    profile_sales: pd.DataFrame
    profile_members: pd.DataFrame


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Data profiling: สำรวจทุกคอลัมน์ (ชนิด ค่าว่าง ค่าไม่ซ้ำ ต่ำสุด สูงสุด ค่าที่พบบ่อย)"""
    rows = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        numeric = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
        top = non_null.value_counts().head(1)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null": int(s.isna().sum()),
            "null_pct": round(s.isna().mean() * 100, 2),
            "unique": int(non_null.nunique()),
            "min": non_null.min() if numeric and len(non_null) else None,
            "max": non_null.max() if numeric and len(non_null) else None,
            "top_value": top.index[0] if len(top) else None,
            "top_pct": round(top.iloc[0] / len(non_null) * 100, 2) if len(top) else None,
        })
    return pd.DataFrame(rows)


def _detail_rows(rule: Rule, violations: pd.Series, ctx: Context) -> pd.DataFrame:
    """แปลงแถวที่ผิดกฎให้เป็นตารางละเอียด ย้อนกลับไปหาแถวต้นฉบับได้"""
    idx = violations[violations].index
    base = pd.DataFrame({"rule_id": rule.rule_id, "dimension": rule.dimension, "table": rule.table}, index=range(len(idx)))
    if rule.table == "sales":
        s = ctx.sales.loc[idx]
        base["row"] = idx
        base["key"] = s["receipt_id"].to_numpy()
        base["branch"] = s["branch"].to_numpy()
        base["source"] = (s["source_file"] + ":" + s["source_row"]).to_numpy()
    elif rule.table == "members":
        m = ctx.members.loc[idx]
        base["row"] = idx
        base["key"] = m["member_id"].to_numpy()
        base["branch"] = m["home_branch"].to_numpy()
        base["source"] = ("members.csv:" + (pd.Series(idx) + 2).astype(str)).to_numpy()
    else:
        base["row"] = None
        base["key"] = [" ".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in idx]
        base["branch"] = [k[0] if isinstance(k, tuple) else "thaphae" if rule.table == "thaphae bills" else None
                          for k in idx]
        base["source"] = None
    return base


def assess(sales: pd.DataFrame, members: pd.DataFrame, products: pd.DataFrame,
           received_at: pd.Timestamp | None = None, extract_report: pd.DataFrame | None = None) -> QualityResult:
    """
    รันกฎคุณภาพทุกข้อแล้วคิดคะแนน (ไม่แก้ข้อมูล)
    ใช้ได้ทั้งกับข้อมูลก่อนทำความสะอาด (โมดูล 3) และหลังทำความสะอาด (โมดูล 4)
    """
    ctx = Context(sales, members, products, received_at if received_at is not None else DEFAULT_RECEIVED_AT)
    rule_rows, details = [], []
    for rule in RULES:
        result = rule.check(ctx)
        if rule.kind == "score":
            rule_rows.append({"rule_id": rule.rule_id, "dimension": rule.dimension, "table": rule.table,
                              "description": rule.description, "checked": len(result),
                              "violations": pd.NA, "score": float(result.mean())})
            continue
        checked, bad = len(result), int(result.sum())
        rule_rows.append({"rule_id": rule.rule_id, "dimension": rule.dimension, "table": rule.table,
                          "description": rule.description, "checked": checked, "violations": bad,
                          "score": 1 - bad / checked if checked else 1.0})
        if bad:
            details.append(_detail_rows(rule, result, ctx))
    rules = pd.DataFrame(rule_rows)
    rules["violations"] = rules["violations"].astype("Int64")
    rules["meets_target"] = rules["score"] >= TARGET

    # ระดับมิติ: Weighted average (น้ำหนักเท่ากัน) + Min operation (Ch3)
    dims = []
    for d in DIMENSIONS:
        r = rules[rules["dimension"] == d]
        worst = r.loc[r["score"].idxmin()]
        dims.append({"dimension": d, "dimension_th": DIMENSION_TH[d], "rules": len(r),
                     "score_avg": r["score"].mean(), "score_min": r["score"].min(),
                     "worst_rule": worst["rule_id"], "meets_target": r["score"].mean() >= TARGET})
    dimensions = pd.DataFrame(dims)

    detail_df = (pd.concat(details, ignore_index=True) if details else
                 pd.DataFrame(columns=["rule_id", "dimension", "table", "row", "key", "branch", "source"]))
    overall = {
        "score_avg": float(dimensions["score_avg"].mean()),
        "score_min": float(dimensions["score_avg"].min()),
        "weakest_dimension": dimensions.loc[dimensions["score_avg"].idxmin(), "dimension"],
        "target": TARGET,
        **summary_metrics(rules, sales, products, extract_report),
    }
    return QualityResult(rules, dimensions, overall, detail_df, by_branch(rules, detail_df, sales),
                         profile(sales), profile(members))


def summary_metrics(rules: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame,
                    extract_report: pd.DataFrame | None) -> dict:
    """ตัวชี้วัดสรุป 4 ตัว จากตาราง 'Example: Measure Data Quality' ใน Ch3"""
    ratio_rules = rules.dropna(subset=["violations"])
    errors, checks = int(ratio_rules["violations"].sum()), int(ratio_rules["checked"].sum())
    empty = int(rules.loc[rules["rule_id"].isin(["CP2", "CP3", "CP4", "CP5", "CP6"]), "violations"].sum())
    if extract_report is not None:
        total = extract_report.set_index("branch").loc["total"]
        fails = int(total[["datetime_unparsed", "product_unmatched", "size_unmapped", "payment_unmapped"]].sum())
        transform_rate = fails / int(total["rows_out"])
    else:
        transform_rate = None
    # Dark data: แถวขายที่นำไปคิดยอดขายไม่ได้เลยถ้าไม่แก้ (ไม่รู้สินค้า / ไม่รู้จำนวน / ไม่รู้ราคา / จำนวน <= 0)
    dark = (sales["product_id"].isna() | sales["qty"].isna() | sales["unit_price"].isna()
            | (sales["qty"] <= 0))
    return {
        "errors": errors, "checks": checks, "error_ratio": errors / checks if checks else 0.0,
        "empty_values": empty,
        "transformation_error_rate": transform_rate,
        "dark_data_rows": int(dark.sum()), "dark_data_pct": float(dark.mean()),
    }


# ---------------------------------------------------------------------------
# ขั้น 3 Analysis: แยกผลตามสาขาเพื่อหาสาเหตุ
# ---------------------------------------------------------------------------
def by_branch(rules: pd.DataFrame, details: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """จำนวนที่ผิดของแต่ละกฎ แยกตามสาขา (ถ้าปัญหาเกิดสาขาเดียว สาเหตุมักอยู่ที่ระบบหรือคนของสาขานั้น)"""
    if details.empty:
        return pd.DataFrame()
    t = details.dropna(subset=["branch"]).pivot_table(index="rule_id", columns="branch", values="dimension",
                                                     aggfunc="count", fill_value=0)
    t = t.reindex(columns=["nimman", "cmu", "thaphae"], fill_value=0)
    t["total"] = t.sum(axis=1)
    order = {r: i for i, r in enumerate(rules["rule_id"])}
    return t.loc[sorted(t.index, key=order.get)].reset_index()


def compare(before: QualityResult, after: QualityResult) -> pd.DataFrame:
    """เทียบคะแนนก่อน / หลังทำความสะอาด (ใช้ในโมดูล 4: Ch3 ขั้น Control, Ch4 ขั้น Validation)"""
    b = before.rules.set_index("rule_id")[["dimension", "description", "violations", "score"]]
    a = after.rules.set_index("rule_id")[["violations", "score"]]
    out = b.join(a, lsuffix="_before", rsuffix="_after")
    out["score_change"] = out["score_after"] - out["score_before"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
def received_at_from_meta(meta: dict | None) -> pd.Timestamp:
    """วันที่รับข้อมูล: ใช้เวลา export ในไฟล์ท่าแพถ้ามี (ไฟล์ CSV / Excel ไม่มีข้อมูลนี้)"""
    if meta and meta.get("exported_at"):
        return pd.Timestamp(meta["exported_at"]).tz_convert(extract.TZ)
    return DEFAULT_RECEIVED_AT


def run_quality(save: bool = True) -> tuple[QualityResult, dict]:
    ext = extract.run_extract(save=False)
    result = assess(ext["sales"], ext["members"], ext["products"],
                    received_at=received_at_from_meta(ext["thaphae_meta"]), extract_report=ext["report"])
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        result.rules.to_csv(PROCESSED_DIR / "quality_rules_before.csv", index=False, encoding="utf-8-sig")
        result.dimensions.to_csv(PROCESSED_DIR / "quality_dimensions_before.csv", index=False, encoding="utf-8-sig")
        result.details.to_csv(PROCESSED_DIR / "quality_details_before.csv", index=False, encoding="utf-8-sig")
        result.profile_sales.to_csv(PROCESSED_DIR / "profile_sales.csv", index=False, encoding="utf-8-sig")
    return result, ext


def _pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


if __name__ == "__main__":
    res, _ = run_quality()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    r = res.rules.assign(score=res.rules["score"].map(_pct))
    print("=== คะแนนรายกฎ ===")
    print(r[["rule_id", "dimension", "table", "checked", "violations", "score", "meets_target", "description"]]
          .to_string(index=False))
    d = res.dimensions.assign(score_avg=res.dimensions["score_avg"].map(_pct), score_min=res.dimensions["score_min"].map(_pct))
    print("\n=== คะแนนรายมิติ (เป้าหมาย", _pct(TARGET), ") ===")
    print(d.to_string(index=False))
    o = res.overall
    print(f"\nคะแนนรวม (Weighted average): {_pct(o['score_avg'])} | มิติที่อ่อนที่สุด (Min): "
          f"{o['weakest_dimension']} {_pct(o['score_min'])}")
    print(f"Ratio of data to errors: {o['errors']:,} / {o['checks']:,} = {_pct(o['error_ratio'])}")
    print(f"Number of empty values: {o['empty_values']:,}")
    print(f"Data transformation error rate: {_pct(o['transformation_error_rate'])}")
    print(f"Dark data: {o['dark_data_rows']:,} แถว ({_pct(o['dark_data_pct'])})")
    print("\n=== แยกตามสาขา (Analysis) ===")
    print(res.by_branch.to_string(index=False))
