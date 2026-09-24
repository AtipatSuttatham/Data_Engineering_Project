"""
โมดูล 4: ทำความสะอาดข้อมูล (Data Cleaning)

"รักษาตามผลตรวจ" แก้ปัญหากลุ่ม B, C ที่โมดูล 3 วัดได้ แล้วตรวจซ้ำด้วยกฎชุดเดิม (Ch4 ขั้น Validation)

หลักการ 3 ข้อ
    1. ใช้ความจริงก่อนใช้สถิติ: ถ้าคำนวณค่าที่ถูกจากข้อมูลอื่นได้ (เช่น ราคาจากตารางราคา) ทำแบบนั้นก่อน
    2. แก้เฉพาะสิ่งที่มั่นใจ ถ้าไม่มั่นใจให้ตัดออกหรือติดธง ไม่เดามั่ว
    3. ทุกการแก้ต้องบันทึก (cleaning log + clean_flags รายแถว) ใช้ทำรายงาน (Ch4 ขั้น Reporting)

ลำดับการทำงาน (Ch4 ขั้น Workflow)
    สมาชิก: M1 ทำข้อความเป็นมาตรฐาน -> M2 รวมสมาชิกซ้ำ -> M3 แก้ปีเกิดผิดกฎ -> M4 เพศที่ว่าง
    ขาย   : S1 ลบข้อมูลซ้ำ -> S2 แก้ชื่อเมนูสะกดผิด -> S3 แก้รหัสสมาชิก -> S4 ค่าผิดกฎ
            -> S5 ตรวจจับ outlier -> S6 เติมค่าว่าง -> S7 คำนวณราคารวมใหม่
    สมาชิก: M5 เติมปีเกิด (ใช้พฤติกรรมการซื้อจากตารางขายที่สะอาดแล้ว) + M6 ติดป้ายค่าที่เติม
    ตรวจซ้ำ: เรียก quality.assess() กับข้อมูลที่สะอาดแล้ว เทียบก่อน / หลัง

วิธีรัน:  py -3.13 -m pipeline.clean
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from pipeline import extract, quality

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SEED = 42

MAX_QTY_PER_LINE = 50        # กฎธุรกิจ: ร้านรับออเดอร์ได้สูงสุด 50 แก้วต่อรายการ
Z_THRESHOLD = 3.0            # Z-score (Ch4 แนะนำ 2.5, 3 หรือ 3.5)
DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 3, 10
IFOREST_CONTAMINATION = 0.005
HOLDOUT_FRACTION = 0.2       # ซ่อนอายุที่รู้ 20% เพื่อวัดความแม่นของวิธีเติมค่า
REFERENCE_YEAR = quality.DEFAULT_RECEIVED_AT.year
THAI_TONE_MARKS = "่้๊๋"
CLEAN_SALES_COLUMNS = extract.STAGING_COLUMNS + ["time_suspect", "clean_flags"]


# ---------------------------------------------------------------------------
# บันทึกการทำความสะอาด (Ch4 ขั้น Reporting)
# ---------------------------------------------------------------------------
@dataclass
class CleaningLog:
    rows: list[dict] = field(default_factory=list)

    def add(self, step: str, table: str, problem: str, technique: str, action: str,
            rows: int, chapter: str, rule: str = "") -> None:
        self.rows.append({"step": step, "table": table, "quality_rule": rule, "problem": problem,
                          "technique": technique, "action": action, "rows": int(rows), "chapter": chapter})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _add_flag(df: pd.DataFrame, mask: pd.Series, name: str) -> None:
    """ติดธงรายแถวใน clean_flags (แถวเดียวมีได้หลายธง คั่นด้วย ;)"""
    m = mask.reindex(df.index, fill_value=False).astype(bool)
    df.loc[m, "clean_flags"] = (df.loc[m, "clean_flags"] + ";" + name).str.strip(";")


def _norm_text(s: pd.Series) -> pd.Series:
    """ทำให้ข้อความเหมือนกันก่อนเทียบ: ตัดช่องว่าง ตัวพิมพ์เล็ก ตัดวรรณยุกต์ไทย"""
    out = s.fillna("").astype(str).str.lower().str.replace(r"\s+", "", regex=True)
    return out.str.replace(f"[{THAI_TONE_MARKS}]", "", regex=True)


# ---------------------------------------------------------------------------
# ตารางสมาชิก M1-M4
# ---------------------------------------------------------------------------
def standardize_members(members: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """M1: ชื่อตัดช่องว่างเกิน / เบอร์โทรเป็นรูปแบบ 0XXXXXXXXX (แก้ CS3)"""
    m = members.copy()
    name = m["full_name"].fillna("").str.split().str.join(" ")
    log.add("M1", "members", "ชื่อมีช่องว่างเกิน", "Standardize", "แก้",
            (name != m["full_name"]).sum(), "Ch4 (Standardize)")
    m["full_name"] = name

    digits = m["phone"].fillna("").str.replace(r"\D", "", regex=True)
    phone = digits.str.replace(r"^66(\d{9})$", r"0\1", regex=True)
    ok = phone.str.fullmatch(r"0\d{9}")
    phone = phone.where(ok, m["phone"])  # รูปแบบที่ไม่รู้จักเก็บค่าเดิมไว้ ไม่เดา
    log.add("M1", "members", "เบอร์โทรหลายรูปแบบ", "Standardize", "แก้",
            (phone != m["phone"]).sum(), "Ch3 Consistency, Ch4", "CS3")
    m["phone"] = phone
    return m


def merge_duplicate_members(members: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, dict]:
    """
    M2: คนเดียวกันสมัครซ้ำ (ชื่อ + เบอร์ตรงกัน) เก็บรหัสที่สมัครก่อนเป็นตัวหลัก (แก้ UQ4)
    ถ้าตัวหลักว่างแต่ตัวซ้ำมีข้อมูล ให้เติมจากตัวซ้ำ
    คืน (ตารางสมาชิก, ตารางแปลงรหัส {รหัสซ้ำ: รหัสหลัก}) ให้ตารางขายใช้ต่อ
    """
    m = members.copy()
    m["_key"] = m["full_name"] + "|" + m["phone"]
    ordered = m.sort_values(["register_date", "member_id"])
    masters = ordered.drop_duplicates("_key", keep="first")
    dups = ordered[~ordered.index.isin(masters.index)]
    master_id = masters.set_index("_key")["member_id"]
    id_map = dict(zip(dups["member_id"], dups["_key"].map(master_id)))

    # groupby().first() เลือกค่าแรกที่ไม่ว่างของแต่ละคน (เรียงตามวันสมัคร)
    coalesced = ordered.groupby("_key")[["gender", "birth_year"]].first()
    out = masters.copy()
    filled = 0
    for c in ["gender", "birth_year"]:
        new = out["_key"].map(coalesced[c])
        filled += int((out[c].isna() & new.notna()).sum())
        out[c] = new.astype(out[c].dtype)
    log.add("M2", "members", "สมาชิกสมัครซ้ำ", "Deduplication (กฎ ชื่อ + เบอร์โทร ต้องไม่ซ้ำ)",
            "รวมเป็นคนเดียว", len(dups), "Ch3 Uniqueness, Lab 5", "UQ4")
    if filled:
        log.add("M2", "members", "ค่าว่างที่เติมได้จากแถวซ้ำ", "Coalesce", "เติม", filled, "Ch3")
    return out.drop(columns="_key").sort_index(), id_map


def fix_birth_year(members: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """M3: ปีเกิดที่เป็น พ.ศ. แก้โดยลบ 543 / ที่เป็นไปไม่ได้ตั้งเป็นว่าง แล้วให้ M5 เติม (แก้ VL4)"""
    m = members.copy()
    by = m["birth_year"]
    valid = (REFERENCE_YEAR - by).between(quality.AGE_MIN, quality.AGE_MAX)
    invalid = by.notna() & ~valid
    be_fixed = by - 543
    is_be = invalid & (REFERENCE_YEAR - be_fixed).between(quality.AGE_MIN, quality.AGE_MAX)
    m.loc[is_be, "birth_year"] = be_fixed[is_be]
    to_na = invalid & ~is_be
    m.loc[to_na, "birth_year"] = pd.NA
    log.add("M3", "members", "ปีเกิดเป็น พ.ศ.", "Validity rule", "แก้ (ลบ 543)", is_be.sum(), "Ch3 Validity", "VL4")
    log.add("M3", "members", "ปีเกิดเป็นไปไม่ได้", "Validity rule", "ตั้งเป็นว่างแล้วเติมใน M5", to_na.sum(),
            "Ch3 Validity", "VL4")
    return m


def fill_gender(members: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """
    M4: เพศที่ว่าง -> "U" (ไม่ระบุ)
    ไม่ใช้ Mode imputation เพราะเพศเดาจากข้อมูลอื่นไม่ได้ ถ้าเติม "F" (พบบ่อยสุด) ทั้งหมด
    ผู้ชายทุกคนในกลุ่มนี้จะถูกเติมผิด และสัดส่วนเพศในรายงานจะเพี้ยน
    """
    m = members.copy()
    missing = m["gender"].isna()
    m.loc[missing, "gender"] = quality.GENDER_UNKNOWN
    log.add("M4", "members", "เพศว่าง", "แยกเป็นกลุ่ม 'ไม่ระบุ' (U) แทน Mode imputation", "ติดค่า U",
            missing.sum(), "Ch4", "CP5")
    return m


# ---------------------------------------------------------------------------
# ตารางขาย S1-S7
# ---------------------------------------------------------------------------
def remove_duplicates(sales: pd.DataFrame, log: CleaningLog, removed: list) -> pd.DataFrame:
    """S1: ลบแถวที่เหมือนกันทุกคอลัมน์ (ไม่นับตำแหน่งในไฟล์) เก็บแถวแรก (แก้ UQ1, UQ2)"""
    core = [c for c in extract.STAGING_COLUMNS if c != "source_row"]
    dup = sales.duplicated(subset=core, keep="first")
    removed.append(sales[dup].assign(removed_reason="duplicate"))
    log.add("S1", "sales", "แถวซ้ำ / บิลซ้ำ", "Deduplication (drop_duplicates)", "ตัดออก", dup.sum(),
            "Ch3 Uniqueness, Ch4, Lab 5", "UQ1, UQ2")
    return sales[~dup].copy()


def build_synonyms(products: pd.DataFrame) -> dict:
    """
    ตารางคำพ้อง: ชื่อไทย / ชื่ออังกฤษ / รหัส -> product_id (หลังทำให้เหมือนกันด้วย _norm_text)
    ถ้าคำพ้องของ 2 สินค้าชนกัน จะไม่ใช้คำนั้น ป้องกันการจับคู่ผิดเมนู
    """
    pairs = []
    for col in ["name_th", "name_en", "nimman_code", "thaphae_sku"]:
        keys = _norm_text(products[col])
        pairs += list(zip(keys, products["product_id"]))
    by_key: dict[str, set] = {}
    for k, pid in pairs:
        by_key.setdefault(k, set()).add(pid)
    return {k: next(iter(v)) for k, v in by_key.items() if len(v) == 1 and k}


def fix_product_names(sales: pd.DataFrame, products: pd.DataFrame, log: CleaningLog,
                      removed: list) -> pd.DataFrame:
    """S2: ชื่อเมนูสะกดผิด -> จับคู่ผ่านตารางคำพ้อง (แก้ CS1)"""
    s = sales.copy()
    unmatched = s["product_id"].isna() & s["product_raw"].notna()
    found = _norm_text(s.loc[unmatched, "product_raw"]).map(build_synonyms(products))
    s.loc[found.index, "product_id"] = found
    fixed = unmatched & s["product_id"].notna()
    _add_flag(s, fixed, "product_synonym")
    log.add("S2", "sales", "ชื่อเมนูสะกดผิด", "Standardize + Synonym table", "แก้", fixed.sum(),
            "Ch2 Mapping, Ch3 Consistency", "CS1")
    still = s["product_id"].isna()
    if still.any():  # ยังไม่รู้ว่าเป็นสินค้าอะไร คิดยอดขายไม่ได้
        removed.append(s[still].assign(removed_reason="unknown_product"))
        log.add("S2", "sales", "ระบุสินค้าไม่ได้", "-", "ตัดออก", still.sum(), "Ch4", "CS1")
    return s[~still].copy()


def fix_member_ids(sales: pd.DataFrame, members: pd.DataFrame, id_map: dict, log: CleaningLog) -> pd.DataFrame:
    """
    S3: รหัสสมาชิกในบิล (แก้ CS2)
      - พิมพ์ตัว O แทนเลข 0 (MO042) -> แก้ถ้ารหัสที่ได้มีอยู่จริง
      - รหัสที่ไม่มีอยู่จริงและแก้ไม่ได้ -> ตั้งเป็นว่าง (ลูกค้าทั่วไป)
      - รหัสของสมาชิกที่สมัครซ้ำ -> แปลงเป็นรหัสหลัก (จาก M2)
    """
    s = sales.copy()
    known = set(members["member_id"]) | set(id_map)
    mid = s["member_id"]
    unknown = mid.notna() & ~mid.isin(known)
    repaired = mid.where(~unknown, mid.str.replace(r"^MO(\d{3})$", r"M0\1", regex=True))
    fixed = unknown & repaired.isin(known)
    to_na = unknown & ~fixed
    s["member_id"] = repaired.where(~to_na, np.nan)
    _add_flag(s, fixed, "member_id_fixed")
    _add_flag(s, to_na, "member_id_unknown")
    merged = s["member_id"].isin(id_map)
    s["member_id"] = s["member_id"].replace(id_map)
    _add_flag(s, merged, "member_id_merged")
    log.add("S3", "sales", "รหัสสมาชิกพิมพ์ O แทน 0", "Pattern repair + ตรวจกับตารางสมาชิก", "แก้",
            fixed.sum(), "Ch3 Consistency", "CS2")
    log.add("S3", "sales", "รหัสสมาชิกไม่มีอยู่จริง", "Consistency ระหว่างตาราง", "ตั้งเป็นว่าง (ลูกค้าทั่วไป)",
            to_na.sum(), "Ch3 Consistency", "CS2")
    log.add("S3", "sales", "ใช้รหัสของสมาชิกที่สมัครซ้ำ", "แปลงเป็นรหัสหลัก", "แก้", merged.sum(),
            "Ch3 Uniqueness", "UQ4")
    return s


def apply_validity_rules(sales: pd.DataFrame, log: CleaningLog, removed: list) -> pd.DataFrame:
    """
    S4: จำนวน <= 0 -> ตัดออก (0 = ไม่มีการขาย / ติดลบ = การคืนสินค้า ไม่นับเป็นยอดขาย) (แก้ VL1)
        เวลานอกเวลาเปิดร้าน -> เก็บไว้ + ติดธง time_suspect เพราะยอดขายจริงแต่เวลาผิด (VL2)
    """
    s = sales.copy()
    bad_qty = s["qty"] <= 0
    removed.append(s[bad_qty].assign(removed_reason="non_positive_qty"))
    log.add("S4", "sales", "จำนวน = 0 หรือติดลบ", "Validity rule", "ตัดออก", bad_qty.sum(), "Ch3 Validity", "VL1")
    s = s[~bad_qty].copy()
    h = s["sale_datetime"].dt.hour
    s["time_suspect"] = (h < quality.OPEN_HOUR) | (h >= quality.CLOSE_HOUR)
    _add_flag(s, s["time_suspect"], "time_suspect")
    log.add("S4", "sales", "เวลานอกเวลาเปิดร้าน", "Validity rule", "เก็บไว้ + ติดธง time_suspect",
            s["time_suspect"].sum(), "Ch3 Validity", "VL2")
    return s


def fix_prices_from_list(sales: pd.DataFrame, products: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """S5 (ราคา): ราคาไม่ตรงตารางราคา (พิมพ์ผิด) -> แก้เป็นราคาตามตาราง เพราะมีค่าอ้างอิง (แก้ AC1)"""
    s = sales.copy()
    exp = quality.expected_price(s, products)
    wrong = s["unit_price"].notna() & exp.notna() & ((s["unit_price"] - exp).abs() > quality.PRICE_TOLERANCE)
    s.loc[wrong, "unit_price"] = exp[wrong]
    _add_flag(s, wrong, "price_fixed_from_list")
    log.add("S5", "sales", "ราคาพิมพ์ผิด", "เทียบค่าอ้างอิง (ตารางราคา)", "แก้เป็นราคาตามตาราง", wrong.sum(),
            "Ch3 Accuracy, Ch4 Outlier", "AC1")
    return s


def outlier_features(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """ตัวแปรที่ใช้ตรวจ outlier ของจำนวน (เฉพาะแถวที่รู้จำนวน)"""
    s = sales[sales["qty"].notna()]
    ref_price = quality.expected_price(s, products)
    ref_price = ref_price.fillna(s["product_id"].map(products.set_index("product_id")["price_M"]).astype(float))
    return pd.DataFrame({
        "qty": s["qty"],
        "value": s["qty"] * ref_price,                      # มูลค่าตามราคาอ้างอิง (ไม่โดนราคาพิมพ์ผิด)
        "items_in_bill": s.groupby("receipt_id")["receipt_id"].transform("size"),
        "hour": s["sale_datetime"].dt.hour,
    }, index=s.index)


def detect_outliers(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    S5 (จำนวน): ตรวจ outlier ด้วย 4 วิธีจาก Ch4 / Ch6 แล้วคืนผลรายแถว (True = outlier)
      Z-score          : |z| > 3 (Ch4, Lab 7)
      IQR              : นอกช่วง Q1 - 1.5 IQR ถึง Q3 + 1.5 IQR (Ch6, Lab 8)
      DBSCAN           : จุดที่ไม่อยู่ในกลุ่มใด (noise) (Ch4, Lab 7)
      Isolation Forest : ใช้หลายตัวแปร (จำนวน มูลค่า จำนวนสินค้าในบิล ชั่วโมง) (Ch4, Lab 7)
    """
    f = outlier_features(sales, products)
    q = f["qty"]
    z = (q - q.mean()) / q.std()
    q1, q3 = q.quantile(0.25), q.quantile(0.75)
    iqr = q3 - q1
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(q.to_frame())
    iso = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=SEED).fit(f)
    return pd.DataFrame({
        "qty": q,
        "zscore": z.abs() > Z_THRESHOLD,
        "iqr": (q < q1 - 1.5 * iqr) | (q > q3 + 1.5 * iqr),
        "dbscan": db.labels_ == -1,
        "isolation_forest": iso.predict(f) == -1,
    }, index=f.index)


def summarize_outliers(flags: pd.DataFrame) -> pd.DataFrame:
    """สรุปผลแต่ละวิธี แยกว่าที่จับได้เกินหรือไม่เกินเกณฑ์ธุรกิจ 50 แก้ว"""
    rows = []
    over = flags["qty"] > MAX_QTY_PER_LINE
    for method in ["zscore", "iqr", "dbscan", "isolation_forest"]:
        m = flags[method]
        rows.append({"method": method, "flagged": int(m.sum()),
                     f"flagged_qty_over_{MAX_QTY_PER_LINE}": int((m & over).sum()),
                     f"flagged_qty_2_to_{MAX_QTY_PER_LINE}": int((m & ~over & (flags["qty"] > 1)).sum()),
                     "missed_qty_over_limit": int((~m & over).sum())})
    return pd.DataFrame(rows)


def remove_qty_typos(sales: pd.DataFrame, flags: pd.DataFrame, log: CleaningLog, removed: list) -> pd.DataFrame:
    """
    ตัดสินใจ: สถิติ (Z-score) ว่าแปลก และ เกินกฎธุรกิจ 50 แก้ว -> พิมพ์ผิด -> ตัดออก
    แปลกแต่ไม่เกิน 50 แก้ว -> ออเดอร์ใหญ่จริง (Natural outlier) -> เก็บไว้
    ตัดออกแทนการหาร 100 เพราะไม่รู้ค่าจริง (100 อาจมาจาก 1 หรือ 10)
    """
    typo = (flags["zscore"] & (flags["qty"] > MAX_QTY_PER_LINE)).reindex(sales.index, fill_value=False)
    natural = (flags["zscore"] & (flags["qty"] <= MAX_QTY_PER_LINE)).reindex(sales.index, fill_value=False)
    # กันกรณีมีข้อผิดพลาดจำนวนมากจน std สูง ทำให้ Z-score ไม่ว่าแปลก (masking) -> ติดธงให้คนตรวจ
    review = ((~flags["zscore"]) & (flags["qty"] > MAX_QTY_PER_LINE)).reindex(sales.index, fill_value=False)
    s = sales.copy()
    _add_flag(s, natural, "large_order_kept")
    _add_flag(s, review, "qty_over_limit_review")
    if review.any():
        log.add("S5", "sales", f"เกิน {MAX_QTY_PER_LINE} แก้วแต่สถิติไม่ว่าแปลก", "ติดธงให้คนตรวจ",
                "เก็บไว้ + ติดธง", review.sum(), "Ch4 Outlier")
    removed.append(s[typo].assign(removed_reason="qty_typo"))
    log.add("S5", "sales", "จำนวนพิมพ์ผิด", f"Z-score + กฎธุรกิจ (> {MAX_QTY_PER_LINE} แก้ว)", "ตัดออก",
            typo.sum(), "Ch4 Outlier")
    log.add("S5", "sales", "ออเดอร์ใหญ่จริง (Natural outlier)", f"Z-score + กฎธุรกิจ (<= {MAX_QTY_PER_LINE} แก้ว)",
            "เก็บไว้ + ติดธง", natural.sum(), "Ch4 Outlier")
    return s[~typo].copy()


def fill_missing_sales(sales: pd.DataFrame, products: pd.DataFrame, log: CleaningLog,
                       removed: list) -> pd.DataFrame:
    """
    S6: เติมค่าว่าง โดยใช้ความจริงก่อนสถิติ (แก้ CP2, CP3, CP4)
      1) ขนาดแก้วว่าง + รู้ราคา  -> ย้อนหาขนาดจากราคา (แต่ละขนาดราคาไม่ซ้ำกัน)
      2) ขนาดแก้วว่าง + ไม่รู้ราคา -> Mode ของเมนูนั้น (สำรองไว้เผื่อ)
      3) จำนวนว่าง (มช.)         -> ราคารวม ÷ ราคาตามตาราง
      4) ราคาต่อชิ้นว่าง          -> ราคาตามตาราง (Enrichment)
    """
    s = sales.copy()
    p = products.set_index("product_id")
    drink = s["product_id"].map(p["category"]).isin(["coffee", "non_coffee"])

    # 1) ขนาดจากราคา
    need_size = drink & s["size"].isna()
    from_price = pd.Series(np.nan, index=s.index, dtype=object)
    for size in ["S", "M", "L"]:
        list_price = s["product_id"].map(p[f"price_{size}"]).astype(float)
        match = need_size & ((s["unit_price"] - list_price).abs() <= quality.PRICE_TOLERANCE)
        from_price[match] = size
    s.loc[from_price.notna(), "size"] = from_price[from_price.notna()]
    _add_flag(s, from_price.notna(), "size_from_price")
    log.add("S6", "sales", "ขนาดแก้วว่าง", "ย้อนหาขนาดจากราคา (ใช้ความจริงก่อนสถิติ)", "เติม (ค่าจริง)",
            from_price.notna().sum(), "Ch2 Attribute construction", "CP4")

    # 2) ขนาดที่ยังว่าง (ไม่รู้ราคา) -> Mode ของเมนูนั้น (ค่าประมาณ ใช้เมื่อไม่มีทางอื่น)
    still = drink & s["size"].isna()
    if still.any():
        mode = s[drink & s["size"].notna()].groupby("product_id")["size"].agg(lambda x: x.mode().iloc[0])
        s.loc[still, "size"] = s.loc[still, "product_id"].map(mode)
        _add_flag(s, still, "size_mode")
    log.add("S6", "sales", "ขนาดแก้วว่างและไม่รู้ราคา", "Mode imputation", "เติม (ค่าประมาณ)", still.sum(),
            "Ch4, Lab 6", "CP4")

    # 3) จำนวนจากราคารวม (ต้องได้จำนวนเต็มบวก ถ้าไม่ได้ถือว่าเชื่อไม่ได้ ตัดออก)
    exp = quality.expected_price(s, products)
    need_qty = s["qty"].isna()
    qty_calc = s["line_total"] / exp
    ok = need_qty & qty_calc.notna() & (qty_calc > 0) & ((qty_calc - qty_calc.round()).abs() < 1e-9)
    s.loc[ok, "qty"] = qty_calc[ok].round()
    _add_flag(s, ok, "qty_from_total")
    log.add("S6", "sales", "จำนวนว่าง", "ราคารวม ÷ ราคาตามตาราง", "เติม (ค่าจริง)", ok.sum(),
            "Ch2 Attribute construction", "CP3")

    # 4) ราคาต่อชิ้นจากตารางราคา
    need_price = s["unit_price"].isna() & exp.notna()
    s.loc[need_price, "unit_price"] = exp[need_price]
    _add_flag(s, need_price, "price_from_list")
    log.add("S6", "sales", "ราคาต่อชิ้นว่าง", "ดึงราคาจากตารางราคา (Enrichment)", "เติม (ค่าจริง)",
            need_price.sum(), "Ch2 Enrichment", "CP2")


    # แถวที่ยังขาดจำนวนหรือราคา คิดยอดขายไม่ได้ -> ตัดออก
    unusable = s["qty"].isna() | s["unit_price"].isna()
    if unusable.any():
        removed.append(s[unusable].assign(removed_reason="cannot_fill"))
        log.add("S6", "sales", "เติมค่าไม่ได้", "-", "ตัดออก", unusable.sum(), "Ch4 Listwise deletion")
    s = s[~unusable].copy()

    # S7: ราคารวม = จำนวน × ราคาต่อชิ้น
    new_total = s["qty"] * s["unit_price"]
    changed = (new_total - s["line_total"]).abs().gt(quality.PRICE_TOLERANCE) | s["line_total"].isna()
    s["line_total"] = new_total
    log.add("S7", "sales", "ราคารวมไม่ตรงหลังแก้ค่า", "คำนวณใหม่ = จำนวน × ราคาต่อชิ้น", "แก้", changed.sum(),
            "Ch2 Attribute construction")
    return s


# ---------------------------------------------------------------------------
# M5-M6 เติมปีเกิดด้วยพฤติกรรมการซื้อ
# ---------------------------------------------------------------------------
def member_features(members: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """ตัวแปรที่ใช้ทายอายุ: สาขาประจำ + พฤติกรรมการซื้อ (สมาชิกที่ไม่เคยซื้อได้ค่า 0)"""
    s = sales[sales["member_id"].notna()].copy()
    s["is_cmu"] = s["branch"] == "cmu"
    s["is_non_coffee"] = s["product_id"].map(products.set_index("product_id")["category"]) == "non_coffee"
    s["hour"] = s["sale_datetime"].dt.hour
    s["weekend"] = s["sale_datetime"].dt.dayofweek >= 5
    bills = s.groupby(["member_id", "receipt_id"]).agg(
        is_cmu=("is_cmu", "first"), hour=("hour", "first"), weekend=("weekend", "first"),
        value=("line_total", "sum"), non_coffee=("is_non_coffee", "mean"))
    g = bills.groupby("member_id")
    beh = pd.DataFrame({
        "n_bills": g.size(), "share_cmu": g["is_cmu"].mean(), "avg_hour": g["hour"].mean(),
        "share_weekend": g["weekend"].mean(), "avg_bill_value": g["value"].mean(),
        "share_non_coffee": g["non_coffee"].mean(),
    })
    f = members[["member_id", "home_branch"]].set_index("member_id")
    # Dummy encoding สาขาประจำ (Ch2) ใช้ N-1 คอลัมน์
    X = pd.get_dummies(f["home_branch"], prefix="home", drop_first=True).astype(float)
    X = X.join(beh).fillna(0.0)
    return X


def compare_imputation(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, str]:
    """
    เทียบวิธีเติมค่าว่างจาก Ch4 ด้วย Hold-out validation:
    ซ่อนอายุที่รู้ไว้ 20% ให้แต่ละวิธีทาย แล้ววัดว่าคลาดเฉลี่ยกี่ปี (MAE) ไม่ต้องใช้ไฟล์เฉลย
    """
    rng = np.random.default_rng(SEED)
    idx = y.index.to_numpy()
    test = rng.choice(idx, size=int(round(len(idx) * HOLDOUT_FRACTION)), replace=False)
    train = np.setdiff1d(idx, test)
    y_tr, y_te = y.loc[train].astype(float), y.loc[test].astype(float)
    reg = LinearRegression().fit(X.loc[train], y_tr)
    pred_reg = reg.predict(X.loc[test])
    resid_sd = float(np.std(y_tr - reg.predict(X.loc[train]), ddof=X.shape[1] + 1))
    preds = {
        "mean": np.full(len(test), y_tr.mean()),
        "median": np.full(len(test), y_tr.median()),
        "regression": pred_reg,
        "stochastic_regression": pred_reg + rng.normal(0, resid_sd, len(test)),
    }
    rows = [{"method": "listwise_deletion", "mae_years": np.nan, "std_after": np.nan,
             "note": f"ตัดสมาชิกที่ไม่มีอายุทิ้ง (ใช้เป็นตัวเทียบ ไม่เติมค่า)"}]
    for name, p in preds.items():
        rows.append({"method": name, "mae_years": float(np.mean(np.abs(p - y_te))),
                     "std_after": float(np.std(np.concatenate([y_tr.to_numpy(), p]))), "note": ""})
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"method": "(ข้อมูลจริง)", "mae_years": np.nan, "std_after": float(y.astype(float).std(ddof=0)),
                             "note": "ส่วนเบี่ยงเบนมาตรฐานของอายุที่รู้ ใช้เทียบว่าวิธีไหนรักษาการกระจายไว้ได้"}
    best = table.dropna(subset=["mae_years"]).sort_values("mae_years").iloc[0]["method"]
    table["chosen"] = table["method"] == best
    return table, best


def impute_birth_year(members: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame,
                      log: CleaningLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M5 + M6: เติมปีเกิดด้วยวิธีที่ทายแม่นที่สุด แล้วติดป้าย birth_year_imputed (Dummy variable adjustment)"""
    m = members.copy()
    X = member_features(m, sales, products)
    y = m.set_index("member_id")["birth_year"]
    known = y.notna()
    table, best = compare_imputation(X[known], y[known])

    missing_ids = y[~known].index
    y_known = y[known].astype(float)
    if best == "mean":
        fill = pd.Series(y_known.mean(), index=missing_ids)
    elif best == "median":
        fill = pd.Series(y_known.median(), index=missing_ids)
    else:
        reg = LinearRegression().fit(X[known], y_known)
        fill = pd.Series(reg.predict(X.loc[missing_ids]), index=missing_ids)
        if best == "stochastic_regression":
            rng = np.random.default_rng(SEED)
            sd = float(np.std(y_known - reg.predict(X[known]), ddof=X.shape[1] + 1))
            fill += rng.normal(0, sd, len(fill))
    lo, hi = REFERENCE_YEAR - quality.AGE_MAX, REFERENCE_YEAR - quality.AGE_MIN
    fill = fill.round().clip(lo, hi).astype(int)

    m["birth_year_imputed"] = m["member_id"].isin(missing_ids)
    m.loc[m["birth_year_imputed"], "birth_year"] = m.loc[m["birth_year_imputed"], "member_id"].map(fill).to_numpy()
    m["birth_year"] = m["birth_year"].astype("Int64")
    log.add("M5", "members", "ปีเกิดว่าง", f"เทียบ 5 วิธี (Hold-out) แล้วใช้ {best}", "เติม (ค่าประมาณ)",
            len(missing_ids), "Ch4 Handle missing data, Lab 6", "CP6")
    log.add("M6", "members", "แยกค่าที่เติมออกจากค่าจริง", "Dummy variable adjustment (birth_year_imputed)",
            "ติดป้าย", len(missing_ids), "Ch4")
    return m, table


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
def run_clean(save: bool = True, ext: dict | None = None) -> dict:
    """
    ทำความสะอาดทั้งหมดตามลำดับ แล้วตรวจซ้ำด้วยโมดูล 3
    คืนค่า dict: sales, members, products, holidays, log, removed, outlier_flags, outlier_summary,
                imputation, member_id_map, quality_before, quality_after, quality_compare
    """
    ext = ext or extract.run_extract(save=False)
    received_at = quality.received_at_from_meta(ext["thaphae_meta"])
    before = quality.assess(ext["sales"], ext["members"], ext["products"],
                            received_at=received_at, extract_report=ext["report"])
    log, removed = CleaningLog(), []
    products = ext["products"]

    # สมาชิก M1-M4
    members = standardize_members(ext["members"], log)
    members, id_map = merge_duplicate_members(members, log)
    members = fix_birth_year(members, log)
    members = fill_gender(members, log)

    # ขาย S1-S7
    sales = ext["sales"].copy()
    sales["clean_flags"] = ""
    sales = remove_duplicates(sales, log, removed)
    sales = fix_product_names(sales, products, log, removed)
    sales = fix_member_ids(sales, members, id_map, log)
    sales = apply_validity_rules(sales, log, removed)
    sales = fix_prices_from_list(sales, products, log)
    outlier_flags = detect_outliers(sales, products)
    sales = remove_qty_typos(sales, outlier_flags, log, removed)
    sales = fill_missing_sales(sales, products, log, removed)
    sales = sales[CLEAN_SALES_COLUMNS].reset_index(drop=True)

    # สมาชิก M5-M6 (ต้องใช้ตารางขายที่สะอาดแล้ว)
    members, imputation = impute_birth_year(members, sales, products, log)
    members = members.reset_index(drop=True)

    after = quality.assess(sales, members, products, received_at=received_at)
    result = {
        "sales": sales, "members": members, "products": products, "holidays": ext["holidays"],
        "log": log.to_frame(),
        "removed": pd.concat(removed, ignore_index=True) if removed else pd.DataFrame(),
        "outlier_flags": outlier_flags, "outlier_summary": summarize_outliers(outlier_flags),
        "imputation": imputation, "member_id_map": id_map,
        "quality_before": before, "quality_after": after, "quality_compare": quality.compare(before, after),
    }
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name in ["sales", "members", "log", "removed", "outlier_summary", "imputation", "quality_compare"]:
            fname = {"sales": "sales_clean", "members": "members_clean", "log": "cleaning_log",
                     "removed": "removed_rows"}.get(name, name)
            result[name].to_csv(PROCESSED_DIR / f"{fname}.csv", index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    res = run_clean()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 50)
    print("=== บันทึกการทำความสะอาด ===")
    print(res["log"][["step", "quality_rule", "problem", "technique", "action", "rows"]].to_string(index=False))
    print("\n=== เปรียบเทียบวิธีตรวจ outlier (จำนวนแก้ว) ===")
    print(res["outlier_summary"].to_string(index=False))
    print("\n=== เปรียบเทียบวิธีเติมปีเกิด (Hold-out MAE) ===")
    print(res["imputation"].to_string(index=False))
    d0, d1 = res["quality_before"].dimensions, res["quality_after"].dimensions
    cmp = d0[["dimension", "score_avg"]].merge(d1[["dimension", "score_avg"]], on="dimension",
                                               suffixes=("_before", "_after"))
    print("\n=== คะแนนคุณภาพ ก่อน / หลัง ===")
    print(cmp.assign(score_avg_before=cmp["score_avg_before"].map("{:.2%}".format),
                     score_avg_after=cmp["score_avg_after"].map("{:.2%}".format)).to_string(index=False))
    print(f"คะแนนรวม: {res['quality_before'].overall['score_avg']:.2%} -> {res['quality_after'].overall['score_avg']:.2%}")
    print(f"\nแถวขาย: {len(res['sales']):,} แถว | ตัดออก {len(res['removed']):,} แถว | สมาชิก {len(res['members']):,} คน")
