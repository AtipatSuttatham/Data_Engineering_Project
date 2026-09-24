"""
โมดูล 2: Extract + Integration

อ่านไฟล์ขายของ 3 สาขาที่รูปแบบต่างกัน แล้วรวมเป็นตารางเดียว (Staging table)
ตามขั้นตอน ETL ใน Ch1: Extract -> Staging Area -> Transform -> Load

สิ่งที่โมดูลนี้ทำ (แก้ปัญหากลุ่ม A = ต่างกันทั้งไฟล์อย่างเป็นระบบ)
    A1 ชื่อคอลัมน์ต่างกัน         -> Schema mapping (Ch5, แนวคิด Global-as-View)
    A2 วันที่ 3 รูปแบบ + UTC      -> แปลงเป็น ISO 8601 เวลาไทย (Ch1, Ch2)
    A3 รหัสสินค้า 3 ระบบ          -> Translation mapping ผ่าน products.csv (Ch2)
    A4 ขนาดเขียนต่างกัน           -> Translation mapping (Ch2)
    A5 ราคาต่อชิ้น vs ราคารวม      -> Attribute construction (Ch2)
    A6 JSON ซ้อนชั้น              -> แตก JSON เป็นตาราง (Ch2)
    A7 ช่องทางชำระเขียนต่างกัน     -> Translation mapping (Ch2)

สิ่งที่โมดูลนี้ "ไม่ทำ" (ปัญหากลุ่ม B, C ปล่อยให้โมดูล 3 วัด และโมดูล 4 แก้)
    ไม่ลบข้อมูลซ้ำ ไม่เติมค่าว่าง ไม่แก้ค่าผิด ถ้าแปลงค่าไม่ได้จะปล่อยว่าง
    เก็บค่าดิบไว้ในคอลัมน์ *_raw และติดธงในคอลัมน์ extract_flags

วิธีรัน:  py -3.13 -m pipeline.extract
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TZ = "Asia/Bangkok"  # เวลาไทย UTC+7

FILES = {
    "nimman": "nimman_sales.csv",
    "cmu": "cmu_sales.xlsx",
    "thaphae": "thaphae_sales.json",
}

# ---------------------------------------------------------------------------
# โครงสร้างกลาง (Global schema) ของตาราง Staging
# ---------------------------------------------------------------------------
STAGING_COLUMNS = [
    # ข้อมูลหลัก
    "branch", "receipt_id", "sale_datetime", "product_id", "size",
    "qty", "unit_price", "line_total", "member_id", "payment",
    # ข้อมูลช่วยตรวจสอบ (lineage): ย้อนกลับไปหาแถวต้นฉบับได้ และใช้แสดง Before / After
    "price_source", "extract_flags",
    "datetime_raw", "product_raw", "size_raw", "payment_raw",
    "source_file", "source_row",
]

# A1: Schema mapping ชื่อคอลัมน์ของแต่ละสาขา -> ชื่อกลาง
# (แนวคิด Global-as-View ใน Ch5: นิยามคอลัมน์กลางจากคอลัมน์ของแต่ละแหล่งข้อมูล)
SCHEMA_MAP = {
    "nimman": {
        "receipt_no": "receipt_id", "sale_datetime": "datetime_raw", "item_code": "product_raw",
        "size": "size_raw", "qty": "qty", "unit_price": "unit_price",
        "member_id": "member_id", "payment": "payment_raw",
    },
    "cmu": {
        "เลขที่บิล": "receipt_id", "ชื่อเมนู": "product_raw", "ขนาด": "size_raw",
        "จำนวน": "qty", "ราคารวม": "line_total", "รหัสสมาชิก": "member_id", "ชำระโดย": "payment_raw",
        # "วันที่" + "เวลา" ต้องรวมกันก่อน จึงจัดการแยกใน standardize_cmu()
    },
    "thaphae": {
        "receipt_id": "receipt_id", "timestamp": "datetime_raw", "sku": "product_raw",
        "size": "size_raw", "quantity": "qty", "price": "unit_price",
        "customer.member_id": "member_id", "payment.method": "payment_raw",
    },
}

# A4, A7: Translation mapping คำศัพท์ของแต่ละสาขา -> ค่ามาตรฐาน
# แยกตามสาขาเพราะแต่ละ POS ใช้คำของตัวเอง ถ้าเจอคำที่ไม่อยู่ในตาราง จะปล่อยว่างและติดธง
SIZE_MAP = {
    "nimman": {"S": "S", "M": "M", "L": "L"},
    "cmu": {"เล็ก": "S", "กลาง": "M", "ใหญ่": "L"},
    "thaphae": {"Small": "S", "Medium": "M", "Large": "L"},
}
PAYMENT_MAP = {
    "nimman": {"CASH": "cash", "QR": "qr", "CARD": "card"},
    "cmu": {"เงินสด": "cash", "พร้อมเพย์": "qr", "บัตรเครดิต": "card"},
    "thaphae": {"cash": "cash", "promptpay": "qr", "credit_card": "card"},
}
# A3: คอลัมน์ใน products.csv ที่ใช้เป็นกุญแจของแต่ละสาขา
PRODUCT_KEY = {"nimman": "nimman_code", "cmu": "name_th", "thaphae": "thaphae_sku"}


# ---------------------------------------------------------------------------
# ① Extract: อ่านไฟล์ (อ่านเป็นข้อความก่อน กันข้อมูลเพี้ยน เช่น เลข 0 นำหน้าหาย)
# ---------------------------------------------------------------------------
def read_nimman(path: Path) -> pd.DataFrame:
    """สาขานิมมาน: CSV 1 แถว = 1 รายการสินค้า"""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df["source_row"] = (df.index + 2).astype(str)  # +2 เพราะแถวที่ 1 ของไฟล์เป็นหัวตาราง
    return df


def read_cmu(path: Path) -> pd.DataFrame:
    """สาขา มช.: Excel หัวคอลัมน์ภาษาไทย (เลขที่บิล 13 หลักต้องอ่านเป็นข้อความ)"""
    df = pd.read_excel(path, dtype={"เลขที่บิล": str, "เวลา": str, "รหัสสมาชิก": str})
    df["source_row"] = (df.index + 2).astype(str)
    return df


def read_thaphae(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    สาขาท่าแพ: JSON ซ้อนชั้น (1 บิลมีหลายสินค้า)
    A6: แตกเป็นตาราง 1 แถว = 1 สินค้า ด้วย pd.json_normalize
        ข้อมูลระดับบิล (เวลา สมาชิก ชำระเงิน) จะถูกคัดลอกไปทุกแถวของบิลนั้น
    คืนค่า (ตาราง, ข้อมูลกำกับไฟล์ เช่น เวลาที่ export) ข้อมูลกำกับใช้วัด Timeliness ในโมดูล 3
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipts = payload["receipts"]
    df = pd.json_normalize(
        receipts,
        record_path="items",
        meta=["receipt_id", "timestamp", ["customer", "member_id"], ["payment", "method"]],
        errors="ignore",
    )
    # ตำแหน่งในไฟล์เดิม เช่น receipts[12].items[1]
    df["source_row"] = [f"receipts[{i}].items[{j}]"
                        for i, r in enumerate(receipts) for j in range(len(r["items"]))]
    meta = {k: v for k, v in payload.items() if k != "receipts"}
    meta["receipts"] = len(receipts)
    return df, meta


def read_members(path: Path) -> pd.DataFrame:
    """ข้อมูลสมาชิก: เบอร์โทรต้องเป็นข้อความ ถ้าอ่านเป็นตัวเลข 0812345678 จะเหลือ 812345678"""
    return pd.read_csv(
        path, encoding="utf-8-sig",
        dtype={"member_id": str, "full_name": str, "gender": str, "phone": str,
               "register_date": str, "home_branch": str, "birth_year": "Int64"},
    )


def read_products(path: Path) -> pd.DataFrame:
    """ตารางสินค้าหลัก (Master data) ใช้เชื่อมรหัสสินค้าทั้ง 3 ระบบ"""
    return pd.read_csv(
        path, encoding="utf-8-sig",
        dtype={"price_S": "Int64", "price_M": "Int64", "price_L": "Int64", "price_nosize": "Int64"},
    )


def read_holidays(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])


# ---------------------------------------------------------------------------
# ②③④ แปลงแต่ละสาขาให้เป็นโครงสร้างกลาง
# ---------------------------------------------------------------------------
def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _map_codes(raw: pd.Series, mapping: dict) -> pd.Series:
    """แปลงค่าตามตาราง ถ้าไม่เจอในตารางจะได้ค่าว่าง (ไม่เดา)"""
    return raw.map(mapping)


def _finish(df: pd.DataFrame, branch: str, products: pd.DataFrame) -> pd.DataFrame:
    """ขั้นตอนที่ทุกสาขาทำเหมือนกัน: แปลงรหัสสินค้า ขนาด ชำระเงิน ติดป้ายที่มา และติดธง"""
    out = df.copy()
    out["branch"] = branch
    out["source_file"] = FILES[branch]

    # A3: รหัสสินค้าของสาขา -> product_id กลาง (จับคู่แบบตรงตัวเท่านั้น)
    key = products.set_index(PRODUCT_KEY[branch])["product_id"]
    out["product_id"] = _map_codes(out["product_raw"], key.to_dict())
    # A4, A7: ขนาดและช่องทางชำระ -> ค่ามาตรฐาน
    out["size"] = _map_codes(out["size_raw"], SIZE_MAP[branch])
    out["payment"] = _map_codes(out["payment_raw"], PAYMENT_MAP[branch])

    # ติดธงแถวที่แปลงไม่ได้ (มีค่าดิบแต่แปลงแล้วว่าง) ให้โมดูล 3 นับ และโมดูล 4 แก้
    flags = pd.DataFrame({
        "datetime_unparsed": out["datetime_raw"].notna() & out["sale_datetime"].isna(),
        "product_unmatched": out["product_raw"].notna() & out["product_id"].isna(),
        "size_unmapped": out["size_raw"].notna() & out["size"].isna(),
        "payment_unmapped": out["payment_raw"].notna() & out["payment"].isna(),
    })
    out["extract_flags"] = flags.apply(lambda r: ";".join(c for c in flags.columns if r[c]), axis=1)
    return out[STAGING_COLUMNS]


def standardize_nimman(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=SCHEMA_MAP["nimman"])  # A1

    # A2: "15/01/2569 08:32" (วว/ดด/พ.ศ.) -> ลบ 543 ปี -> เวลาไทย
    parts = df["datetime_raw"].str.extract(r"^(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2})$")
    dt = pd.to_datetime(
        dict(year=_to_number(parts[2]) - 543, month=_to_number(parts[1]), day=_to_number(parts[0]),
             hour=_to_number(parts[3]), minute=_to_number(parts[4])),
        errors="coerce",
    )
    df["sale_datetime"] = dt.dt.tz_localize(TZ)

    # A5: มีราคาต่อชิ้นในไฟล์ -> คำนวณราคารวม
    df["qty"] = _to_number(df["qty"])
    df["unit_price"] = _to_number(df["unit_price"])
    df["line_total"] = df["qty"] * df["unit_price"]
    df["price_source"] = df["unit_price"].notna().map({True: "file", False: pd.NA})
    return _finish(df, "nimman", products)


def standardize_cmu(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=SCHEMA_MAP["cmu"])  # A1

    # A2: วันที่และเวลาอยู่คนละคอลัมน์ (ปี ค.ศ.) -> รวมเป็นค่าเดียว -> เวลาไทย
    date_text = pd.to_datetime(df["วันที่"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["datetime_raw"] = date_text + " " + df["เวลา"]
    df["sale_datetime"] = pd.to_datetime(df["datetime_raw"], format="%Y-%m-%d %H:%M",
                                         errors="coerce").dt.tz_localize(TZ)

    # A5: ไฟล์มีแต่ราคารวม -> คำนวณราคาต่อชิ้น = ราคารวม ÷ จำนวน (Attribute construction)
    # ถ้าจำนวนว่างหรือเป็น 0 จะหารไม่ได้ ปล่อยว่างไว้ให้โมดูล 4
    df["qty"] = _to_number(df["qty"])
    df["line_total"] = _to_number(df["line_total"])
    can_derive = df["qty"].notna() & (df["qty"] != 0) & df["line_total"].notna()
    df["unit_price"] = (df["line_total"] / df["qty"]).where(can_derive)
    df["price_source"] = can_derive.map({True: "derived_from_total", False: pd.NA})
    return _finish(df, "cmu", products)


def standardize_thaphae(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=SCHEMA_MAP["thaphae"])  # A1
    # ค่า null ใน JSON อ่านมาเป็น None ส่วนไฟล์อื่นเป็น NaN ทำให้เป็นแบบเดียวกัน
    for c in ["size_raw", "member_id"]:
        df[c] = df[c].where(df[c].notna(), np.nan)

    # A2: "2026-01-15T01:32:00Z" เป็นเวลา UTC -> แปลงเป็นเวลาไทย (บวก 7 ชั่วโมง)
    df["sale_datetime"] = pd.to_datetime(df["datetime_raw"], format="%Y-%m-%dT%H:%M:%SZ",
                                         utc=True, errors="coerce").dt.tz_convert(TZ)

    # A5: มีราคาต่อชิ้น -> คำนวณราคารวม
    df["qty"] = _to_number(df["qty"])
    df["unit_price"] = _to_number(df["unit_price"])
    df["line_total"] = df["qty"] * df["unit_price"]
    df["price_source"] = df["unit_price"].notna().map({True: "file", False: pd.NA})
    return _finish(df, "thaphae", products)


# ---------------------------------------------------------------------------
# ⑤⑥ Integration + ตรวจยอด
# ---------------------------------------------------------------------------
def integrate(parts: dict[str, pd.DataFrame], rows_read: dict[str, int]) -> pd.DataFrame:
    """
    ต่อตารางทุกสาขาเป็นตารางเดียว (append rows แบบ Lab 2)
    แล้วตรวจยอด: จำนวนแถวที่ได้ต้องเท่ากับจำนวนแถวที่อ่านเข้า (Load verification ใน Ch1)
    """
    staging = pd.concat(list(parts.values()), ignore_index=True)
    expected = sum(rows_read.values())
    if len(staging) != expected:
        raise ValueError(f"ตรวจยอดไม่ผ่าน: อ่านเข้า {expected} แถว แต่ได้ {len(staging)} แถว")
    return staging


def build_report(parts: dict[str, pd.DataFrame], rows_read: dict[str, int],
                 thaphae_meta: dict) -> pd.DataFrame:
    """รายงานการ Extract รายสาขา: อ่านได้กี่แถว และแปลงค่าอะไรไม่ได้กี่แถว"""
    rows = []
    for branch, df in parts.items():
        flags = df["extract_flags"]
        rows.append({
            "branch": branch,
            "source_file": FILES[branch],
            "records_in_file": thaphae_meta["receipts"] if branch == "thaphae" else rows_read[branch],
            "rows_read": rows_read[branch],
            "rows_out": len(df),
            "datetime_unparsed": int(flags.str.contains("datetime_unparsed").sum()),
            "product_unmatched": int(flags.str.contains("product_unmatched").sum()),
            "size_unmapped": int(flags.str.contains("size_unmapped").sum()),
            "payment_unmapped": int(flags.str.contains("payment_unmapped").sum()),
        })
    report = pd.DataFrame(rows)
    total = report.drop(columns=["branch", "source_file"]).sum()
    # ระเบียนในไฟล์นับคนละหน่วย (แถว vs บิล) จึงไม่รวมยอดในแถว total
    total["records_in_file"] = pd.NA
    report.loc[len(report)] = {"branch": "total", "source_file": "-", **total.to_dict()}
    count_cols = report.columns.drop(["branch", "source_file"])
    report[count_cols] = report[count_cols].astype("Int64")
    report["reconciled"] = report["rows_read"] == report["rows_out"]
    return report


def run_extract(raw_dir: Path = RAW_DIR, save: bool = True) -> dict:
    """
    เรียกทุกขั้นตอนของโมดูล 2
    คืนค่า dict: sales (ตาราง staging), members, products, holidays, report, thaphae_meta
    """
    products = read_products(raw_dir / "products.csv")
    nimman_raw = read_nimman(raw_dir / FILES["nimman"])
    cmu_raw = read_cmu(raw_dir / FILES["cmu"])
    thaphae_raw, thaphae_meta = read_thaphae(raw_dir / FILES["thaphae"])

    rows_read = {"nimman": len(nimman_raw), "cmu": len(cmu_raw), "thaphae": len(thaphae_raw)}
    parts = {
        "nimman": standardize_nimman(nimman_raw, products),
        "cmu": standardize_cmu(cmu_raw, products),
        "thaphae": standardize_thaphae(thaphae_raw, products),
    }
    sales = integrate(parts, rows_read)
    result = {
        "sales": sales,
        "members": read_members(raw_dir / "members.csv"),
        "products": products,
        "holidays": read_holidays(raw_dir / "holidays.csv"),
        "report": build_report(parts, rows_read, thaphae_meta),
        "thaphae_meta": thaphae_meta,
    }
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        sales.to_csv(PROCESSED_DIR / "sales_staging.csv", index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    res = run_extract()
    pd.set_option("display.width", 200)
    print("รายงานการ Extract")
    print(res["report"].to_string(index=False))
    print(f"\nบันทึกตาราง staging {len(res['sales']):,} แถว ที่ data/processed/sales_staging.csv")
