"""
โมดูล 1: สร้างข้อมูลสมมุติของร้านกาแฟ "Doi Brew" 3 สาขา (นิมมาน, มช., ท่าแพ)

ขั้นตอนการทำงาน
    ขั้น 1  สร้าง Master data         -> products.csv, holidays.csv
    ขั้น 2  สร้างข้อมูลสมาชิกแบบสะอาด
    ขั้น 3  จำลองการขายแบบสะอาด       (ยอดขายขึ้นกับสาขา วันในสัปดาห์ วันหยุด และช่วงปิดเทอม)
    ขั้น 4  ใส่ปัญหาคุณภาพข้อมูล (กลุ่ม B) และ outlier (กลุ่ม C) แล้วจดทุกรายการลงไฟล์เฉลย
    ขั้น 5  แปลงข้อมูลเป็นรูปแบบของแต่ละสาขา (CSV / Excel / JSON) ซึ่งทำให้เกิดปัญหากลุ่ม A

วิธีรัน:  py -3.13 data_generator/generate.py
ใช้ SEED คงที่ รันกี่ครั้งก็ได้ข้อมูลเหมือนเดิม
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ค่าตั้งต้น (ปรับได้ที่นี่ที่เดียว)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SEED = 42

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 6, 30)
N_MEMBERS = 800
OPEN_HOUR, CLOSE_HOUR = 7, 20  # ร้านเปิด 07:00 - 20:00
CSV_ENCODING = "utf-8-sig"  # มี BOM เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน (pandas อ่านได้ปกติ)

BRANCHES = ["nimman", "cmu", "thaphae"]
BASE_BILLS_PER_DAY = {"nimman": 25, "cmu": 24, "thaphae": 19}
MEMBER_BILL_RATE = {"nimman": 0.40, "cmu": 0.50, "thaphae": 0.25}
PAYMENT_PROB = {  # สัดส่วนช่องทางชำระเงิน: cash, qr, card
    "nimman": [0.25, 0.55, 0.20],
    "cmu": [0.35, 0.60, 0.05],
    "thaphae": [0.40, 0.25, 0.35],
}

# สมมุติฐานสำหรับการจำลอง (ไม่ใช่ปฏิทินจริงของมหาวิทยาลัย)
CMU_BREAK = (date(2026, 3, 21), date(2026, 5, 31))
SONGKRAN = (date(2026, 4, 13), date(2026, 4, 15))
THAPHAE_OFFLINE_DAYS = [date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 27)]

# สัดส่วนปัญหาที่ใส่ลงในข้อมูล (อ้างอิงรหัสปัญหาใน docs/data_issues.md)
ISSUE_RATES = {
    "B1_duplicate_rows_nimman": 0.015,    # สัดส่วนของรายการสินค้าสาขานิมมาน
    "B2_duplicate_bills_thaphae": 40,     # จำนวนบิล
    "B3_duplicate_members": 0.02,         # สัดส่วนของสมาชิก
    "B4_missing_gender": 0.08,
    "B5_missing_birth_year": 0.10,
    "B6_missing_unit_price_nimman": 0.03,
    "B7_missing_size_drinks": 0.02,
    "B8_missing_qty_cmu": 0.01,
    "B9_invalid_birth_year": 0.01,
    "B10_non_positive_qty": 0.005,
    "B11_out_of_hours_nimman": 0.003,     # สัดส่วนของบิล
    "B12_unknown_member_id": 0.01,        # สัดส่วนของบิลที่มีรหัสสมาชิก
    "B14_menu_name_variants_cmu": 0.03,
    "C1_qty_typo": 0.003,
    "C2_price_typo": 0.002,
    "C3_natural_big_orders": 20,          # จำนวนบิล (ไม่ใช่ข้อผิดพลาด)
}

# ---------------------------------------------------------------------------
# Master data
# ---------------------------------------------------------------------------
# product_id, name_th, name_en, category, nimman_code, thaphae_sku, S, M, L, ไม่มีขนาด, ความนิยม
PRODUCTS = [
    ("P01", "เอสเปรสโซ่", "Espresso", "coffee", "C01", "DB-COF-ESP", 45, 55, 65, None, 6),
    ("P02", "อเมริกาโน่", "Americano", "coffee", "C02", "DB-COF-AME", 45, 55, 65, None, 14),
    ("P03", "ลาเต้", "Latte", "coffee", "C03", "DB-COF-LAT", 55, 65, 75, None, 16),
    ("P04", "คาปูชิโน่", "Cappuccino", "coffee", "C04", "DB-COF-CAP", 55, 65, 75, None, 9),
    ("P05", "มอคค่า", "Mocha", "coffee", "C05", "DB-COF-MOC", 60, 70, 80, None, 7),
    ("P06", "คาราเมลมัคคิอาโต้", "Caramel Macchiato", "coffee", "C06", "DB-COF-CMA", 65, 75, 85, None, 8),
    ("P07", "ชาไทย", "Thai Tea", "non_coffee", "T01", "DB-TEA-THA", 45, 55, 65, None, 14),
    ("P08", "มัทฉะลาเต้", "Matcha Latte", "non_coffee", "T02", "DB-TEA-MAT", 60, 70, 80, None, 10),
    ("P09", "ชามะนาว", "Lemon Tea", "non_coffee", "T03", "DB-TEA-LEM", 40, 50, 60, None, 6),
    ("P10", "โกโก้", "Cocoa", "non_coffee", "T04", "DB-OTH-COC", 50, 60, 70, None, 7),
    ("P11", "นมสดคาราเมล", "Caramel Fresh Milk", "non_coffee", "T05", "DB-OTH-CFM", 50, 60, 70, None, 5),
    ("P12", "ครัวซองต์", "Croissant", "bakery", "B01", "DB-BAK-CRO", None, None, None, 55, 8),
    ("P13", "บราวนี่", "Brownie", "bakery", "B02", "DB-BAK-BRO", None, None, None, 50, 6),
    ("P14", "เค้กช็อกโกแลต", "Chocolate Cake", "bakery", "B03", "DB-BAK-CHC", None, None, None, 85, 5),
    ("P15", "แซนด์วิชแฮมชีส", "Ham Cheese Sandwich", "bakery", "B04", "DB-BAK-HCS", None, None, None, 75, 5),
]

# คู่เมนูที่มักซื้อด้วยกัน (ให้โมเดล Recommendation ค้นพบในโมดูล 7)
PAIRS = {"P02": "P12", "P07": "P13", "P03": "P14", "P01": "P15"}
PAIR_PROB = 0.45

# วันหยุดปี 2569 (สมมุติสำหรับการจำลอง วันหยุดทางพุทธศาสนาอิงปฏิทินจันทรคติ)
HOLIDAYS = [
    ("2026-01-01", "วันขึ้นปีใหม่", "New Year's Day"),
    ("2026-03-03", "วันมาฆบูชา", "Makha Bucha Day"),
    ("2026-04-06", "วันจักรี", "Chakri Memorial Day"),
    ("2026-04-13", "วันสงกรานต์", "Songkran Festival"),
    ("2026-04-14", "วันสงกรานต์", "Songkran Festival"),
    ("2026-04-15", "วันสงกรานต์", "Songkran Festival"),
    ("2026-05-01", "วันแรงงานแห่งชาติ", "National Labour Day"),
    ("2026-05-04", "วันฉัตรมงคล", "Coronation Day"),
    ("2026-05-31", "วันวิสาขบูชา", "Visakha Bucha Day"),
    ("2026-06-01", "ชดเชยวันวิสาขบูชา", "Substitution for Visakha Bucha Day"),
    ("2026-06-03", "วันเฉลิมพระชนมพรรษาพระราชินี", "Queen Suthida's Birthday"),
]

# ชื่อไทยสมมุติ
FIRST_NAMES_M = ["สมชาย", "ธนากร", "กิตติพัฒน์", "ณัฐวุฒิ", "พงศกร", "วรเมธ", "ศุภชัย", "อนุชา",
                 "ภูมิพัฒน์", "ธีรภัทร", "ปกรณ์", "จิรายุ", "ชยพล", "นพดล", "วีรภาพ", "สิทธิชัย",
                 "เอกชัย", "ภาคิน", "กฤษดา", "ธนวัฒน์"]
FIRST_NAMES_F = ["สมหญิง", "ปิยะดา", "ณัฐธิดา", "กมลชนก", "ศิริพร", "อรอุมา", "พิมพ์ชนก", "ชนิดา",
                 "วรรณภา", "ธิดารัตน์", "สุภาวดี", "จิราพร", "ปวีณา", "กัญญาณัฐ", "เบญญาภา", "อัญชิสา",
                 "นันทนา", "รัชนีกร", "พัชรินทร์", "มณีรัตน์"]
LAST_NAMES = ["ใจดี", "ศรีสุข", "บัวบาน", "หลวงแสน", "กันยา", "มอรสิล", "แก้วมณี", "ทองดี",
              "สุขสวัสดิ์", "วงศ์ใหญ่", "ปัญญาดี", "อินทร์แก้ว", "คำมูล", "ไชยวงศ์", "ศรีวิชัย",
              "จันทร์หอม", "ดวงดี", "มีสุข", "พรหมมา", "เรือนคำ", "ธรรมวงศ์", "ยอดเมือง",
              "สุวรรณรัตน์", "บุญมา", "ทาคำ", "อุดมศักดิ์", "เชียงทอง", "ปันแก้ว", "คำแสน", "นันทะ",
              "สายสุข", "วงค์ชัย", "จันทร์แก้ว", "สมบูรณ์", "ศักดิ์ดี", "แสงทอง", "ใจมา", "ปิ่นแก้ว",
              "อินต๊ะ", "กาวิละ"]

# คำศัพท์ที่แต่ละสาขาใช้ (ต้นเหตุของปัญหากลุ่ม A)
SIZE_VOCAB = {
    "nimman": {"S": "S", "M": "M", "L": "L"},
    "cmu": {"S": "เล็ก", "M": "กลาง", "L": "ใหญ่"},
    "thaphae": {"S": "Small", "M": "Medium", "L": "Large"},
}
PAYMENT_VOCAB = {
    "nimman": {"cash": "CASH", "qr": "QR", "card": "CARD"},
    "cmu": {"cash": "เงินสด", "qr": "พร้อมเพย์", "card": "บัตรเครดิต"},
    "thaphae": {"cash": "cash", "qr": "promptpay", "card": "credit_card"},
}

# น้ำหนักจำนวนลูกค้าในแต่ละชั่วโมง 07:00 ถึง 19:00 (เช้าและบ่ายคนเยอะ)
HOUR_WEIGHTS = np.array([6, 10, 12, 10, 7, 6, 6, 9, 9, 7, 5, 4, 3], dtype=float)


class IssueLog:
    """จดบันทึกทุกปัญหาที่ใส่ลงไป (ไฟล์เฉลย) ใช้วัดว่า pipeline จับปัญหาได้ครบแค่ไหน"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, issue_id: str, file: str, **detail) -> None:
        self.records.append({"issue_id": issue_id, "file": file, **detail})

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["issue_id"]] = counts.get(r["issue_id"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (kv[0][0], int(kv[0][1:].split("_")[0]))))


def to_native(value):
    """แปลงค่า numpy ให้เป็นชนิดของ Python เพื่อเขียนลง JSON ได้"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# ขั้น 1: Master data
# ---------------------------------------------------------------------------
def build_products() -> pd.DataFrame:
    cols = ["product_id", "name_th", "name_en", "category", "nimman_code", "thaphae_sku",
            "price_S", "price_M", "price_L", "price_nosize", "popularity"]
    df = pd.DataFrame(PRODUCTS, columns=cols)
    for c in ["price_S", "price_M", "price_L", "price_nosize"]:
        df[c] = df[c].astype("Int64")
    return df


def build_holidays() -> pd.DataFrame:
    return pd.DataFrame(HOLIDAYS, columns=["date", "name_th", "name_en"])


# ---------------------------------------------------------------------------
# ขั้น 2: สมาชิก (แบบสะอาด)
# ---------------------------------------------------------------------------
def build_members(rng: np.random.Generator) -> pd.DataFrame:
    # สุ่มชื่อแบบไม่ซ้ำ (คนละคนต้องชื่อ-นามสกุลไม่ซ้ำกัน) ชื่อที่ซ้ำจึงมีแค่จากการสมัครซ้ำ (B3)
    name_pool = {g: [(f, l) for f in names for l in LAST_NAMES]
                 for g, names in (("M", FIRST_NAMES_M), ("F", FIRST_NAMES_F))}
    for g in name_pool:
        name_pool[g] = [name_pool[g][k] for k in rng.permutation(len(name_pool[g]))]
    rows = []
    for i in range(1, N_MEMBERS + 1):
        home = rng.choice(BRANCHES, p=[0.40, 0.40, 0.20])
        gender = rng.choice(["M", "F"], p=[0.45, 0.55])
        first, last = name_pool[gender].pop()
        # สาขา มช. ลูกค้าส่วนใหญ่เป็นนักศึกษา อายุน้อยกว่าสาขาอื่น
        age = int(np.clip(rng.normal(21, 2), 18, 30)) if home == "cmu" else int(np.clip(rng.normal(34, 9), 18, 65))
        phone_digits = "0" + rng.choice(["8", "9", "6"]) + "".join(str(d) for d in rng.integers(0, 10, 8))
        reg = date(2023, 1, 1) + timedelta(days=int(rng.integers(0, 1095)))
        rows.append({
            "member_id": f"M{i:04d}", "full_name": f"{first} {last}", "gender": gender,
            "birth_year": 2026 - age, "phone": phone_digits,
            "register_date": reg.isoformat(), "home_branch": home,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ขั้น 3: จำลองการขาย (แบบสะอาด)
# ---------------------------------------------------------------------------
def expected_bills(branch: str, d: date, holiday_dates: set[date]) -> float:
    """จำนวนบิลที่คาดว่าจะเกิดในวันนั้น ใส่รูปแบบไว้ให้ Dashboard และ Regression ค้นพบ"""
    lam = BASE_BILLS_PER_DAY[branch]
    weekend = d.weekday() >= 5
    is_holiday = d in holiday_dates
    if branch == "nimman":
        lam *= 1.3 if weekend else 1.0
        lam *= 1.2 if is_holiday else 1.0
    elif branch == "cmu":
        lam *= 0.5 if weekend else 1.15
        lam *= 0.35 if CMU_BREAK[0] <= d <= CMU_BREAK[1] else 1.0
        lam *= 0.4 if is_holiday else 1.0
    else:  # thaphae: high season ม.ค.-ก.พ. แล้วค่อย ๆ ลดลง
        lam *= {1: 1.35, 2: 1.3, 3: 1.0, 4: 0.85, 5: 0.75, 6: 0.75}[d.month]
        lam *= 1.2 if weekend else 1.0
        lam *= 1.3 if is_holiday else 1.0
    if SONGKRAN[0] <= d <= SONGKRAN[1]:
        lam *= {"nimman": 1.3, "cmu": 1.0, "thaphae": 1.8}[branch]
    return lam


def pick_product(rng, hour: int, by_cat: dict[str, pd.DataFrame]) -> str:
    """เลือกเมนูแรกของบิล เช้าคนสั่งกาแฟมาก บ่ายและเย็นสั่งชาและอื่น ๆ มากขึ้น"""
    if hour < 12:
        p = [0.60, 0.25, 0.15]
    elif hour < 17:
        p = [0.40, 0.40, 0.20]
    else:
        p = [0.30, 0.50, 0.20]
    cat = rng.choice(["coffee", "non_coffee", "bakery"], p=p)
    items = by_cat[cat]
    w = items["popularity"].to_numpy(float)
    return str(rng.choice(items["product_id"].to_numpy(), p=w / w.sum()))


def build_bills(rng, products: pd.DataFrame, members: pd.DataFrame, holiday_dates: set[date]) -> list[dict]:
    by_cat = {c: g for c, g in products.groupby("category")}
    drink_ids = set(products.loc[products["category"] != "bakery", "product_id"])
    bakery = by_cat["bakery"]
    members_by_branch = {b: members.loc[members["home_branch"] == b, "member_id"].to_numpy() for b in BRANCHES}
    all_members = members["member_id"].to_numpy()
    hours = np.arange(OPEN_HOUR, CLOSE_HOUR)

    def size_for(pid: str):
        return str(rng.choice(["S", "M", "L"], p=[0.25, 0.50, 0.25])) if pid in drink_ids else None

    def qty() -> int:
        return int(rng.choice([1, 2, 3], p=[0.80, 0.15, 0.05]))

    bills = []
    d = START_DATE
    while d <= END_DATE:
        for branch in BRANCHES:
            n = rng.poisson(expected_bills(branch, d, holiday_dates))
            for _ in range(n):
                hour = int(rng.choice(hours, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum()))
                dt = datetime(d.year, d.month, d.day, hour, int(rng.integers(0, 60)))
                first = pick_product(rng, hour, by_cat)
                items = [(first, size_for(first), qty())]
                if first in PAIRS and rng.random() < PAIR_PROB:
                    items.append((PAIRS[first], None, 1))
                elif first in drink_ids and rng.random() < 0.12:
                    items.append((str(rng.choice(bakery["product_id"].to_numpy())), None, 1))
                if rng.random() < 0.15:
                    extra = pick_product(rng, hour, by_cat)
                    if extra not in [it[0] for it in items]:
                        items.append((extra, size_for(extra), 1))
                member = None
                if rng.random() < MEMBER_BILL_RATE[branch]:
                    pool = members_by_branch[branch] if rng.random() < 0.8 else all_members
                    member = str(rng.choice(pool))
                payment = str(rng.choice(["cash", "qr", "card"], p=PAYMENT_PROB[branch]))
                bills.append({"branch": branch, "sale_dt": dt, "member_id": member,
                              "payment": payment, "items": items, "natural_outlier": False})
        d += timedelta(days=1)
    return bills


def add_natural_big_orders(rng, bills: list[dict], log: IssueLog) -> None:
    """C3: ออเดอร์ใหญ่จริง (เช่น บริษัทสั่งกาแฟไปประชุม) เป็น outlier ที่ถูกต้อง ห้ามลบ"""
    n = ISSUE_RATES["C3_natural_big_orders"]
    branches = rng.choice(BRANCHES, size=n, p=[0.6, 0.2, 0.2])
    for branch in branches:
        while True:  # เลือกวันทำงาน ช่วงเช้า
            d = START_DATE + timedelta(days=int(rng.integers(0, (END_DATE - START_DATE).days + 1)))
            if d.weekday() < 5 and not (branch == "thaphae" and d in THAPHAE_OFFLINE_DAYS):
                break
        dt = datetime(d.year, d.month, d.day, int(rng.integers(8, 10)), int(rng.integers(0, 60)))
        pid = str(rng.choice(["P02", "P03", "P07"]))
        q = int(rng.integers(20, 41))
        bills.append({"branch": str(branch), "sale_dt": dt, "member_id": None, "payment": "card",
                      "items": [(pid, "M", q)], "natural_outlier": True})
    bills.sort(key=lambda b: (b["branch"], b["sale_dt"]))


def assign_receipt_ids(rng, bills: list[dict]) -> None:
    """แต่ละสาขาใช้รูปแบบเลขที่บิลของ POS คนละยี่ห้อ"""
    seq = {"nimman": 0}
    daily: dict[date, int] = {}
    tp_ids = rng.choice(16 ** 6, size=len(bills), replace=False)
    for i, b in enumerate(bills):
        if b["branch"] == "nimman":
            seq["nimman"] += 1
            b["receipt_id"] = f"NM-{seq['nimman']:06d}"
        elif b["branch"] == "cmu":
            day = b["sale_dt"].date()
            daily[day] = daily.get(day, 0) + 1
            b["receipt_id"] = f"{day:%Y%m%d}{daily[day]:05d}"
        else:
            b["receipt_id"] = f"TP-{int(tp_ids[i]):06X}"


def bills_to_lines(bills: list[dict], products: pd.DataFrame) -> pd.DataFrame:
    price = products.set_index("product_id")
    rows = []
    for b in bills:
        for line_no, (pid, size, q) in enumerate(b["items"], start=1):
            p = price.loc[pid]
            unit = p["price_nosize"] if size is None else p[f"price_{size}"]
            rows.append({"branch": b["branch"], "receipt_id": b["receipt_id"], "line_no": line_no,
                         "sale_dt": b["sale_dt"], "product_id": pid, "size": size, "qty": float(q),
                         "unit_price": float(unit), "member_id": b["member_id"],
                         "payment": b["payment"], "natural_outlier": b["natural_outlier"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ขั้น 4: ใส่ปัญหา
# ---------------------------------------------------------------------------
class LinePicker:
    """สุ่มเลือกแถวที่จะใส่ปัญหา โดยแต่ละแถวโดนปัญหาได้แค่ 1 อย่าง อธิบายในเล่มได้ง่ายกว่า"""

    def __init__(self, rng, sales: pd.DataFrame) -> None:
        self.rng = rng
        self.used = sales["natural_outlier"].to_numpy().copy()  # ออเดอร์ใหญ่จริงไม่ต้องใส่ปัญหาเพิ่ม

    def pick(self, mask: np.ndarray, rate: float) -> np.ndarray:
        candidates = np.flatnonzero(mask & ~self.used)
        n = int(round(mask.sum() * rate))
        chosen = np.sort(self.rng.choice(candidates, size=n, replace=False))
        self.used[chosen] = True
        return chosen


def inject_sales_issues(rng, sales: pd.DataFrame, products: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    s = sales.copy()
    s["menu_name_raw"] = s["product_id"].map(products.set_index("product_id")["name_th"])
    names_en = products.set_index("product_id")["name_en"]
    picker = LinePicker(rng, s)
    branch = s["branch"].to_numpy()
    all_rows = np.ones(len(s), dtype=bool)

    def rec(issue, idx, column, before, after):
        row = s.loc[idx]
        log.add(issue, f"{row['branch']}_sales", receipt_id=row["receipt_id"], line_no=int(row["line_no"]),
                column=column, original=to_native(before), injected=to_native(after))

    # B10: จำนวนเป็น 0 หรือติดลบ (บันทึกการคืนเงินผิดช่อง)
    for idx in picker.pick(all_rows, ISSUE_RATES["B10_non_positive_qty"]):
        before = s.at[idx, "qty"]
        after = 0.0 if rng.random() < 0.4 else -before
        s.at[idx, "qty"] = after
        rec("B10", idx, "qty", before, after)

    # C1: จำนวนพิมพ์ผิด กด 0 เกินมา 2 ตัว (1 -> 100)
    for idx in picker.pick(all_rows, ISSUE_RATES["C1_qty_typo"]):
        before = s.at[idx, "qty"]
        s.at[idx, "qty"] = before * 100
        rec("C1", idx, "qty", before, before * 100)

    # C2: ราคาพิมพ์ผิด (55 -> 5500)
    for idx in picker.pick(all_rows, ISSUE_RATES["C2_price_typo"]):
        before = s.at[idx, "unit_price"]
        s.at[idx, "unit_price"] = before * 100
        rec("C2", idx, "unit_price", before, before * 100)

    # สาขา มช. เก็บ "ราคารวม" คำนวณหลังใส่ B10/C1/C2 ความผิดพลาดจึงติดไปถึงราคารวมเหมือนของจริง
    s["line_total"] = s["qty"] * s["unit_price"]

    # B6: ราคาต่อชิ้นว่าง (นิมมาน)
    for idx in picker.pick(branch == "nimman", ISSUE_RATES["B6_missing_unit_price_nimman"]):
        rec("B6", idx, "unit_price", s.at[idx, "unit_price"], None)
        s.at[idx, "unit_price"] = np.nan

    # B7: ขนาดแก้วว่าง (เฉพาะเครื่องดื่ม ส่วนเบเกอรี่ไม่มีขนาดอยู่แล้วซึ่งไม่ใช่ปัญหา)
    for idx in picker.pick(s["size"].notna().to_numpy(), ISSUE_RATES["B7_missing_size_drinks"]):
        rec("B7", idx, "size", s.at[idx, "size"], None)
        s.at[idx, "size"] = None

    # B8: จำนวนว่าง (มช.) คำนวณกลับได้จากราคารวม ÷ ราคาต่อชิ้น
    for idx in picker.pick(branch == "cmu", ISSUE_RATES["B8_missing_qty_cmu"]):
        rec("B8", idx, "qty", s.at[idx, "qty"], None)
        s.at[idx, "qty"] = np.nan

    # B14: ชื่อเมนูสะกดไม่ตรง (มช. พนักงานพิมพ์เอง)
    for idx in picker.pick(branch == "cmu", ISSUE_RATES["B14_menu_name_variants_cmu"]):
        name = s.at[idx, "menu_name_raw"]
        en = names_en[s.at[idx, "product_id"]]
        variants = [name + " ", " " + name, en.upper(), en.lower()]
        if "้" in name:
            variants.append(name.replace("้", ""))  # ลืมใส่ไม้โท
        variant = str(rng.choice(variants))
        s.at[idx, "menu_name_raw"] = variant
        rec("B14", idx, "menu_name", name, variant)

    # ---- ปัญหาระดับบิล (ทุกแถวในบิลเดียวกันโดนพร้อมกัน) ----
    bills = s.drop_duplicates("receipt_id")[["receipt_id", "branch", "member_id", "natural_outlier"]]

    # B11: เวลาขายนอกเวลาเปิดร้าน (นาฬิกา POS ผิด)
    nm_bills = bills[(bills["branch"] == "nimman") & ~bills["natural_outlier"]]["receipt_id"].to_numpy()
    for rid in rng.choice(nm_bills, size=int(round(len(nm_bills) * ISSUE_RATES["B11_out_of_hours_nimman"])), replace=False):
        m = s["receipt_id"] == rid
        before = s.loc[m, "sale_dt"].iloc[0]
        after = before.replace(hour=int(rng.integers(0, 6)))
        s.loc[m, "sale_dt"] = after
        log.add("B11", "nimman_sales", receipt_id=rid, column="sale_datetime",
                original=before.isoformat(), injected=after.isoformat())

    # B12: รหัสสมาชิกที่ไม่มีอยู่จริง (พิมพ์ผิด)
    mem_bills = bills[bills["member_id"].notna()]["receipt_id"].to_numpy()
    for rid in rng.choice(mem_bills, size=int(round(len(mem_bills) * ISSUE_RATES["B12_unknown_member_id"])), replace=False):
        m = s["receipt_id"] == rid
        before = s.loc[m, "member_id"].iloc[0]
        kind = rng.integers(0, 3)
        if kind == 0:
            after = f"M9{rng.integers(0, 1000):03d}"   # รหัสเกินช่วงที่มีจริง
        elif kind == 1:
            after = "MO" + before[2:]                   # พิมพ์ตัว O แทนเลข 0
        else:
            after = "M" + before[2:]                    # ตัวเลขหายไป 1 หลัก
        s.loc[m, "member_id"] = after
        log.add("B12", f"{s.loc[m, 'branch'].iloc[0]}_sales", receipt_id=rid, column="member_id",
                original=before, injected=after)
    return s


def inject_member_issues(rng, members: pd.DataFrame, sales: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = members.copy()
    s = sales.copy()
    n = len(m)
    used = np.zeros(n, dtype=bool)

    def pick(rate):
        cand = np.flatnonzero(~used)
        chosen = np.sort(rng.choice(cand, size=int(round(n * rate)), replace=False))
        used[chosen] = True
        return chosen

    # B9: ปีเกิดเป็นไปไม่ได้ (กรอก พ.ศ. / อายุติดลบ / อายุ 150 ปี)
    for idx in pick(ISSUE_RATES["B9_invalid_birth_year"]):
        before = int(m.at[idx, "birth_year"])
        after = [before + 543, 2031, 1876][int(rng.integers(0, 3))]
        m.at[idx, "birth_year"] = after
        log.add("B9", "members", member_id=m.at[idx, "member_id"], column="birth_year", original=before, injected=after)

    m["birth_year"] = m["birth_year"].astype("Int64")
    # B5: ปีเกิดว่าง
    for idx in pick(ISSUE_RATES["B5_missing_birth_year"]):
        log.add("B5", "members", member_id=m.at[idx, "member_id"], column="birth_year",
                original=int(m.at[idx, "birth_year"]), injected=None)
        m.at[idx, "birth_year"] = pd.NA

    # B4: เพศว่าง (สุ่มแยกจาก B5 จึงมีบางแถวที่ว่างทั้งสองช่อง ซึ่งเกิดได้จริง)
    for idx in np.sort(rng.choice(n, size=int(round(n * ISSUE_RATES["B4_missing_gender"])), replace=False)):
        log.add("B4", "members", member_id=m.at[idx, "member_id"], column="gender",
                original=m.at[idx, "gender"], injected=None)
        m.at[idx, "gender"] = None

    # B3: สมาชิกสมัครซ้ำ ชื่อเว้นวรรคต่างกันแต่เบอร์เดียวกัน สมัครใหม่ระหว่างช่วงข้อมูล
    dup_src = np.sort(rng.choice(n, size=int(round(n * ISSUE_RATES["B3_duplicate_members"])), replace=False))
    new_rows = []
    for k, idx in enumerate(dup_src, start=1):
        row = m.loc[idx].copy()
        new_id = f"M{N_MEMBERS + k:04d}"
        first, last = row["full_name"].split(" ")
        row["full_name"] = [f"{first}  {last}", f"{first} {last} ", f" {first} {last}"][int(rng.integers(0, 3))]
        reg = START_DATE + timedelta(days=int(rng.integers(20, 120)))
        row["register_date"] = reg.isoformat()
        old_id = row["member_id"]
        row["member_id"] = new_id
        new_rows.append(row)
        # หลังสมัครซ้ำ ลูกค้าใช้รหัสใหม่ในบางบิล ต้องรวมกลับเป็นคนเดียวใน pipeline
        target = s[(s["member_id"] == old_id) & (s["sale_dt"].dt.date >= reg)]["receipt_id"].unique()
        switched = [r for r in target if rng.random() < 0.5]
        s.loc[s["receipt_id"].isin(switched), "member_id"] = new_id
        log.add("B3", "members", member_id=new_id, duplicate_of=old_id, column="full_name",
                injected=row["full_name"], bills_using_new_id=len(switched))
    m = pd.concat([m, pd.DataFrame(new_rows)], ignore_index=True)

    # B13: เบอร์โทรหลายรูปแบบ (กรอกอิสระ ทั้งไฟล์)
    def fmt_phone(p: str) -> str:
        style = int(rng.integers(0, 4))
        return [f"{p[:3]}-{p[3:6]}-{p[6:]}", p, f"+66{p[1:]}", f"{p[:3]} {p[3:6]} {p[6:]}"][style]

    m["phone"] = m["phone"].map(fmt_phone)
    log.add("B13", "members", column="phone", rows=len(m), note="ทุกแถวสุ่ม 4 รูปแบบ")
    return m, s


def duplicate_rows(rng, sales: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    """B1: แถวซ้ำทุกคอลัมน์ในสาขานิมมาน (POS ส่งซ้ำตอนเน็ตหลุด) วางไว้ต่อจากแถวเดิม"""
    s = sales.reset_index(drop=True)
    nm_idx = np.flatnonzero(s["branch"].to_numpy() == "nimman")
    chosen = set(rng.choice(nm_idx, size=int(round(len(nm_idx) * ISSUE_RATES["B1_duplicate_rows_nimman"])), replace=False))
    order = []
    for i in range(len(s)):
        order.append(i)
        if i in chosen:
            order.append(i)
            log.add("B1", "nimman_sales", receipt_id=s.at[i, "receipt_id"], line_no=int(s.at[i, "line_no"]))
    return s.iloc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ขั้น 5: แปลงเป็นรูปแบบของแต่ละสาขา
# ---------------------------------------------------------------------------
def write_nimman(s: pd.DataFrame, products: pd.DataFrame, path: Path) -> None:
    code = products.set_index("product_id")["nimman_code"]
    nm = s[s["branch"] == "nimman"]
    out = pd.DataFrame({
        "receipt_no": nm["receipt_id"],
        # วว/ดด/ปี พ.ศ. ชั่วโมง:นาที
        "sale_datetime": nm["sale_dt"].map(lambda d: f"{d.day:02d}/{d.month:02d}/{d.year + 543} {d:%H:%M}"),
        "item_code": nm["product_id"].map(code),
        "size": nm["size"].map(SIZE_VOCAB["nimman"]),
        "qty": nm["qty"].astype("Int64"),
        "unit_price": nm["unit_price"].astype("Int64"),
        "member_id": nm["member_id"],
        "payment": nm["payment"].map(PAYMENT_VOCAB["nimman"]),
    })
    out.to_csv(path, index=False, encoding=CSV_ENCODING)


def write_cmu(s: pd.DataFrame, path: Path) -> None:
    cm = s[s["branch"] == "cmu"]
    out = pd.DataFrame({
        "เลขที่บิล": cm["receipt_id"],
        "วันที่": pd.to_datetime(cm["sale_dt"].dt.date),
        "เวลา": cm["sale_dt"].dt.strftime("%H:%M"),
        "ชื่อเมนู": cm["menu_name_raw"],
        "ขนาด": cm["size"].map(SIZE_VOCAB["cmu"]),
        "จำนวน": cm["qty"].astype("Int64"),
        "ราคารวม": cm["line_total"].astype("Int64"),
        "รหัสสมาชิก": cm["member_id"],
        "ชำระโดย": cm["payment"].map(PAYMENT_VOCAB["cmu"]),
    })
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="YYYY-MM-DD", date_format="YYYY-MM-DD") as xw:
        out.to_excel(xw, index=False, sheet_name="ยอดขาย")


def write_thaphae(rng, s: pd.DataFrame, products: pd.DataFrame, path: Path, log: IssueLog) -> None:
    sku = products.set_index("product_id")["thaphae_sku"]
    tp = s[s["branch"] == "thaphae"]
    bills = []
    for rid, g in tp.groupby("receipt_id", sort=False):
        first = g.iloc[0]
        utc = first["sale_dt"] - timedelta(hours=7)  # เวลาไทย (UTC+7) -> UTC
        bills.append({
            "receipt_id": rid,
            "timestamp": f"{utc:%Y-%m-%dT%H:%M:%S}Z",
            "customer": {"member_id": to_native(first["member_id"])},
            "payment": {"method": PAYMENT_VOCAB["thaphae"][first["payment"]]},
            "items": [{"sku": sku[r["product_id"]],
                       "size": SIZE_VOCAB["thaphae"].get(r["size"]) if r["size"] else None,
                       "quantity": to_native(int(r["qty"])),
                       "price": to_native(int(r["unit_price"]))} for _, r in g.iterrows()],
        })
    bills.sort(key=lambda b: b["timestamp"])
    # B2: บิลซ้ำทั้งก้อน (ระบบ sync ส่งซ้ำ) ต่อท้ายไฟล์
    for i in np.sort(rng.choice(len(bills), size=ISSUE_RATES["B2_duplicate_bills_thaphae"], replace=False)):
        bills.append(json.loads(json.dumps(bills[i], ensure_ascii=False)))
        log.add("B2", "thaphae_sales", receipt_id=bills[i]["receipt_id"])
    payload = {"branch": "Doi Brew Tha Phae", "exported_at": "2026-07-01T00:05:00Z", "receipts": bills}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_members(m: pd.DataFrame, path: Path) -> None:
    m.to_csv(path, index=False, encoding=CSV_ENCODING)


# ---------------------------------------------------------------------------
# เอกสารสรุปปัญหา (สร้างอัตโนมัติ ตัวเลขจึงตรงกับข้อมูลจริงเสมอ)
# ---------------------------------------------------------------------------
ISSUE_CATALOG = [
    # (รหัส, ปัญหา, ไฟล์/คอลัมน์, สาเหตุในโลกจริง, เทคนิคที่จะใช้แก้, บท)
    ("A1", "ชื่อและภาษาของคอลัมน์ไม่ตรงกัน", "ทุกไฟล์ขาย", "POS คนละยี่ห้อ", "Schema mapping", "Ch5"),
    ("A2", "วันที่ 3 รูปแบบ (วว/ดด/พ.ศ., แยกวันกับเวลา, ISO UTC)", "ทุกไฟล์ขาย", "POS คนละยี่ห้อ", "แปลงเป็น ISO 8601 เวลาไทย", "Ch1, Ch2"),
    ("A3", "รหัสสินค้า 3 ระบบ (C03 / ลาเต้ / DB-COF-LAT)", "ทุกไฟล์ขาย", "POS คนละยี่ห้อ", "Translation mapping ผ่าน products.csv", "Ch2"),
    ("A4", "ขนาดเขียนต่างกัน (M / กลาง / Medium)", "ทุกไฟล์ขาย", "POS คนละยี่ห้อ", "Mapping + Ordinal encoding", "Ch2"),
    ("A5", "มช. เก็บราคารวม ไม่ใช่ราคาต่อชิ้น", "cmu_sales.xlsx / ราคารวม", "POS คนละยี่ห้อ", "Attribute construction", "Ch2"),
    ("A6", "JSON ซ้อนหลายชั้น (1 บิลมีหลายสินค้า)", "thaphae_sales.json", "ระบบ export เป็น JSON", "แตก JSON เป็นตาราง", "Ch2"),
    ("A7", "ช่องทางชำระเขียนต่างกัน (QR / พร้อมเพย์ / promptpay)", "ทุกไฟล์ขาย", "POS คนละยี่ห้อ", "Standardize", "Ch3 Consistency"),
    ("B1", "แถวซ้ำทุกคอลัมน์", "nimman_sales.csv", "POS ส่งซ้ำตอนเน็ตหลุด", "Uniqueness + ลบข้อมูลซ้ำ", "Ch3, Lab 5"),
    ("B2", "บิลซ้ำทั้งก้อน", "thaphae_sales.json", "ระบบ sync ส่งซ้ำ", "ตรวจซ้ำด้วย receipt_id", "Ch3"),
    ("B3", "สมาชิกสมัครซ้ำ (ชื่อเว้นวรรคต่างกัน เบอร์เดียวกัน)", "members.csv", "ลูกค้าลืมว่าเคยสมัคร", "กฎ ชื่อ+เบอร์ ต้องไม่ซ้ำ แล้วรวมรหัส", "Ch3"),
    ("B4", "เพศว่าง", "members.csv / gender", "ไม่บังคับกรอก", "Mode หรือแยกเป็น 'ไม่ระบุ'", "Ch4, Lab 6"),
    ("B5", "ปีเกิดว่าง", "members.csv / birth_year", "ไม่บังคับกรอก", "เทียบ Median กับ Regression imputation", "Ch4, Lab 6"),
    ("B6", "ราคาต่อชิ้นว่าง", "nimman_sales.csv / unit_price", "พนักงานลืมกรอก", "เติมจากตารางราคา (Enrichment)", "Ch2"),
    ("B7", "ขนาดแก้วว่าง (เฉพาะเครื่องดื่ม)", "ทุกไฟล์ขาย / size", "กดข้าม", "Mode ของเมนูนั้น", "Ch4"),
    ("B8", "จำนวนว่าง", "cmu_sales.xlsx / จำนวน", "ช่องหาย", "คำนวณกลับ = ราคารวม ÷ ราคาต่อชิ้น", "Ch2"),
    ("B9", "ปีเกิดเป็นไปไม่ได้ (พ.ศ., ปี 2031, ปี 1876)", "members.csv / birth_year", "กรอก พ.ศ. ในช่อง ค.ศ. / พิมพ์ผิด", "Validity rule", "Ch3, Lab 5"),
    ("B10", "จำนวนเป็น 0 หรือติดลบ", "ทุกไฟล์ขาย / qty", "คืนเงินบันทึกผิดช่อง", "Validity rule แล้วตัดออก", "Ch3"),
    ("B11", "เวลาขายนอกเวลาเปิดร้าน (00:00-05:59)", "nimman_sales.csv / sale_datetime", "นาฬิกา POS ผิด", "Validity rule แล้วติดธง", "Ch3"),
    ("B12", "รหัสสมาชิกที่ไม่มีอยู่จริง", "ทุกไฟล์ขาย / member_id", "พิมพ์รหัสผิด", "Consistency ระหว่างตาราง", "Ch3"),
    ("B13", "เบอร์โทรหลายรูปแบบ", "members.csv / phone", "กรอกอิสระ", "Standardize แล้ว Anonymization", "Ch2"),
    ("B14", "ชื่อเมนูสะกดไม่ตรง", "cmu_sales.xlsx / ชื่อเมนู", "พนักงานพิมพ์เอง", "Standardize + ตารางคำพ้อง", "Ch2, Ch3"),
    ("B15", "ข้อมูลหายทั้งวัน", "thaphae_sales.json", "POS ออฟไลน์", "Completeness ของช่วงวันที่ + Timeliness", "Ch3"),
    ("C1", "จำนวนพิมพ์ผิด (×100)", "ทุกไฟล์ขาย / qty", "กด 0 เกิน", "Outlier detection แล้วแก้หรือลบ", "Ch4, Lab 7"),
    ("C2", "ราคาพิมพ์ผิด (×100)", "ทุกไฟล์ขาย / ราคา", "กด 0 เกิน", "Outlier detection แล้วแก้จากตารางราคา", "Ch4, Lab 7"),
    ("C3", "ออเดอร์ใหญ่จริง 20-40 แก้ว (ไม่ใช่ข้อผิดพลาด)", "ทุกไฟล์ขาย", "บริษัทสั่งไปประชุม", "ต้องไม่ลบ (Natural outlier)", "Ch4"),
]


def write_issue_doc(summary: dict[str, int], stats: dict, path: Path) -> None:
    lines = [
        "# รายการปัญหาในข้อมูล Doi Brew",
        "",
        "> ไฟล์นี้สร้างอัตโนมัติจาก `data_generator/generate.py` ตัวเลขจึงตรงกับข้อมูลจริงเสมอ ห้ามแก้ด้วยมือ",
        "> รายละเอียดรายแถวอยู่ใน `data/generator_log/injected_issues.json`",
        "",
        "## ภาพรวมข้อมูล",
        "",
        "| รายการ | จำนวน |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in stats.items()]
    lines += [
        "",
        "## ปัญหาที่ใส่ไว้",
        "",
        "กลุ่ม A เกิดจากรูปแบบไฟล์ของแต่ละสาขาต่างกัน (มีทุกแถว) / กลุ่ม B คือปัญหาคุณภาพข้อมูล / กลุ่ม C คือ outlier",
        "",
        "| รหัส | ปัญหา | ไฟล์ / คอลัมน์ | จำนวนที่ใส่ | สาเหตุในโลกจริง | เทคนิคที่จะใช้แก้ | บท |",
        "|---|---|---|---|---|---|---|",
    ]
    for code, problem, where, cause, tech, ch in ISSUE_CATALOG:
        if code.startswith("A"):
            count = "ทุกแถว"
        elif code == "B13":
            count = "ทุกแถว"
        elif code == "B15":
            count = f"{len(THAPHAE_OFFLINE_DAYS)} วัน ({stats['บิลที่หายไปจาก B15']} บิล)"
        else:
            count = f"{summary.get(code, 0):,}"
        lines.append(f"| {code} | {problem} | {where} | {count} | {cause} | {tech} | {ch} |")
    lines += [
        "",
        "## สมมุติฐานของการจำลอง",
        "",
        f"- ช่วงข้อมูล {START_DATE:%d/%m/%Y} ถึง {END_DATE:%d/%m/%Y} ร้านเปิด {OPEN_HOUR:02d}:00-{CLOSE_HOUR:02d}:00",
        f"- ช่วงปิดเทอม มช. (สมมุติ): {CMU_BREAK[0]:%d/%m/%Y} ถึง {CMU_BREAK[1]:%d/%m/%Y} ยอดขายสาขา มช. ลดเหลือ 35%",
        f"- วันที่ POS ท่าแพออฟไลน์ (B15): {', '.join(f'{d:%d/%m/%Y}' for d in THAPHAE_OFFLINE_DAYS)}",
        "- วันหยุดใน `holidays.csv` เป็นค่าสมมุติสำหรับการจำลอง วันหยุดทางพุทธศาสนาอิงปฏิทินจันทรคติ ควรตรวจกับประกาศจริงถ้าจะนำไปใช้ต่อ",
        f"- คู่เมนูที่มักซื้อด้วยกัน (ให้ Recommendation ค้นพบ): "
        + ", ".join(f"{a}+{b}" for a, b in PAIRS.items()) + f" โอกาส {PAIR_PROB:.0%}",
        "- แต่ละแถวโดนปัญหาได้แค่ 1 อย่าง (ยกเว้น B4 เพศว่างที่สุ่มแยก และปัญหาระดับบิล B11, B12)",
        "- แถวที่สมัครซ้ำ (B3) คัดลอกค่ามาจากแถวเดิมรวมถึงค่าว่าง จำนวนค่าว่างทั้งไฟล์จึงอาจมากกว่าตัวเลข B4/B5 เล็กน้อย",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(out_root: Path = ROOT, verbose: bool = True) -> dict:
    rng = np.random.default_rng(SEED)
    raw = out_root / "data" / "raw"
    gen_log = out_root / "data" / "generator_log"
    docs = out_root / "docs"
    for p in (raw, gen_log, docs):
        p.mkdir(parents=True, exist_ok=True)
    log = IssueLog()

    # ขั้น 1
    products = build_products()
    holidays = build_holidays()
    holiday_dates = {date.fromisoformat(d) for d in holidays["date"]}

    # ขั้น 2 และ 3
    members = build_members(rng)
    bills = build_bills(rng, products, members, holiday_dates)
    add_natural_big_orders(rng, bills, log)

    # B15: POS ท่าแพออฟไลน์ ข้อมูลวันนั้นหายทั้งวัน
    lost = [b for b in bills if b["branch"] == "thaphae" and b["sale_dt"].date() in THAPHAE_OFFLINE_DAYS]
    lost_ids = {id(b) for b in lost}
    bills = [b for b in bills if id(b) not in lost_ids]
    for d in THAPHAE_OFFLINE_DAYS:
        log.add("B15", "thaphae_sales", date=d.isoformat(),
                bills_lost=sum(1 for b in lost if b["sale_dt"].date() == d))

    assign_receipt_ids(rng, bills)
    clean_sales = bills_to_lines(bills, products)
    for b in bills:
        if b["natural_outlier"]:
            log.add("C3", f"{b['branch']}_sales", receipt_id=b["receipt_id"], qty=b["items"][0][2])

    # เก็บข้อมูลสะอาดไว้เป็นเฉลย (ใช้วัดความแม่นของการเติมค่าว่างและการแก้ข้อมูลในโมดูล 4)
    clean_sales.drop(columns="natural_outlier").to_csv(gen_log / "sales_truth.csv", index=False, encoding=CSV_ENCODING)
    members.to_csv(gen_log / "members_truth.csv", index=False, encoding=CSV_ENCODING)

    # ขั้น 4
    sales = inject_sales_issues(rng, clean_sales, products, log)
    members_dirty, sales = inject_member_issues(rng, members, sales, log)
    sales = duplicate_rows(rng, sales, log)

    # ขั้น 5
    write_nimman(sales, products, raw / "nimman_sales.csv")
    write_cmu(sales, raw / "cmu_sales.xlsx")
    write_thaphae(rng, sales, products, raw / "thaphae_sales.json", log)
    write_members(members_dirty, raw / "members.csv")
    products.drop(columns="popularity").to_csv(raw / "products.csv", index=False, encoding=CSV_ENCODING)
    holidays.to_csv(raw / "holidays.csv", index=False, encoding=CSV_ENCODING)

    summary = log.summary()
    (gen_log / "injected_issues.json").write_text(
        json.dumps({"seed": SEED, "summary": summary, "issues": log.records}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    stats = {
        "บิลทั้งหมด (ข้อมูลสะอาด)": len(bills),
        "รายการสินค้าทั้งหมด (ข้อมูลสะอาด)": len(clean_sales),
        **{f"บิลสาขา{name}": sum(1 for b in bills if b["branch"] == key)
           for key, name in [("nimman", "นิมมาน"), ("cmu", " มช."), ("thaphae", "ท่าแพ")]},
        "สมาชิก (ไม่รวมที่สมัครซ้ำ)": N_MEMBERS,
        "บิลที่หายไปจาก B15": len(lost),
    }
    write_issue_doc(summary, stats, docs / "data_issues.md")

    if verbose:
        print("สร้างข้อมูลเสร็จแล้ว")
        for k, v in stats.items():
            print(f"  {k}: {v:,}")
        print("ปัญหาที่ใส่ไว้:", ", ".join(f"{k}={v}" for k, v in summary.items()))
    return {"summary": summary, "stats": stats}


if __name__ == "__main__":
    main()
