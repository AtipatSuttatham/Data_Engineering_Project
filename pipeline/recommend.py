"""
โมดูล 7: ระบบแนะนำเมนู (Recommendation) ด้วย Market Basket Analysis / Association Rules

ตอบคำถาม "ลูกค้าสั่งเมนูนี้ ควรแนะนำเมนูไหนเพิ่ม" โดยนับจากบิลในอดีตว่าเมนูไหนมักอยู่ในบิลเดียวกัน
เป็นโมเดลโบนัส (นอกเนื้อหาวิชา) ส่วนที่เชื่อมกับวิชา:
    - ตาราง basket (บิล × เมนู 0/1) = แนวคิด One-hot / Bag of Words (Ch7)
    - Support / Confidence = การนับความถี่ (Aggregation, Ch2 / สถิติเชิงพรรณนา, Ch6)

ตัวเลข 3 ตัวของกฎ "ซื้อ A -> ซื้อ B"
    Support    = บิลที่มีทั้ง A และ B / บิลทั้งหมด        (เกิดบ่อยแค่ไหน)
    Confidence = บิลที่มีทั้ง A และ B / บิลที่มี A        (ซื้อ A แล้วมีโอกาสซื้อ B กี่ %)
    Lift       = Confidence / สัดส่วนบิลที่มี B          (บ่อยกว่าความบังเอิญกี่เท่า)

ขั้นตอน: อ่าน basket จาก SQLite -> แบ่งตามเวลา (ฝึก ม.ค.-พ.ค. / ทดสอบ มิ.ย.) -> หากฎจากข้อมูลฝึก
         -> วัดผลแบบซ่อน 1 เมนู (Leave-one-out) เทียบกับ Baseline -> บันทึกกฎลง SQLite (reco_rules)

วิธีรัน:  py -3.13 -m pipeline.recommend
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import load

MIN_SUPPORT = 0.01      # คู่นี้ต้องเกิดอย่างน้อย 1% ของบิล (ตัดคู่ที่อาจเป็นความบังเอิญ)
MIN_CONFIDENCE = 0.20   # แนะนำแล้วต้องมีโอกาสขายได้อย่างน้อย 20%
MIN_LIFT = 1.2          # ต้องเกิดบ่อยกว่าความบังเอิญอย่างน้อย 1.2 เท่า
TEST_MONTH = 6          # ข้อมูลทดสอบ = มิ.ย. (ข้อมูลฝึก = เดือนก่อนหน้า)
TOP_K = 3               # แนะนำ 3 เมนู


# ---------------------------------------------------------------------------
# เตรียมข้อมูล
# ---------------------------------------------------------------------------
def load_baskets(db_path: Path = load.DB_PATH) -> pd.DataFrame:
    """อ่านตาราง basket + วันที่ของแต่ละบิล จากฐานข้อมูล (โมดูล 6)"""
    basket = load.read_table("basket", db_path)
    dates = load.query("SELECT receipt_id, MIN(date_key) AS date_key FROM fact_sales GROUP BY receipt_id", db_path)
    out = basket.merge(dates, on="receipt_id", how="left")
    out["month"] = out["date_key"] // 100 % 100
    return out


def item_columns(baskets: pd.DataFrame) -> list[str]:
    return [c for c in baskets.columns if c.startswith("P")]


def split_by_time(baskets: pd.DataFrame, test_month: int = TEST_MONTH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """แบ่งตามเวลา: ใช้อดีตหากฎ แล้วทดสอบกับอนาคต (ตรงกับการใช้งานจริง และกันข้อมูลอนาคตรั่ว)"""
    return baskets[baskets["month"] < test_month].copy(), baskets[baskets["month"] == test_month].copy()


# ---------------------------------------------------------------------------
# หากฎ (Association rules)
# ---------------------------------------------------------------------------
def mine_rules(baskets: pd.DataFrame, min_support: float = MIN_SUPPORT, min_confidence: float = MIN_CONFIDENCE,
               min_lift: float = MIN_LIFT) -> pd.DataFrame:
    """
    คำนวณ Support, Confidence, Lift ของทุกคู่เมนู (A -> B)
    เมนูมีแค่ 15 เมนู (210 กฎ) จึงคำนวณครบทุกคู่ด้วยการคูณเมทริกซ์ ไม่ต้องใช้ Apriori ตัดทิ้งก่อน
    คืนทุกคู่ พร้อมคอลัมน์ passes = ผ่านเกณฑ์ทั้ง 3 ข้อหรือไม่
    """
    items = item_columns(baskets)
    X = baskets[items].to_numpy(dtype=np.int64)
    n = len(X)
    co = X.T @ X                        # co[a, b] = จำนวนบิลที่มีทั้ง a และ b (แนวทแยง = บิลที่มี a)
    item_count = np.diag(co)
    rows = []
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i == j or item_count[i] == 0 or item_count[j] == 0:
                continue
            both = co[i, j]
            support = both / n
            confidence = both / item_count[i]
            lift = confidence / (item_count[j] / n)
            rows.append({"antecedent": a, "consequent": b, "count_ab": int(both), "count_a": int(item_count[i]),
                         "support": support, "confidence": confidence, "lift": lift})
    pairs = pd.DataFrame(rows)
    pairs["passes"] = ((pairs["support"] >= min_support) & (pairs["confidence"] >= min_confidence)
                       & (pairs["lift"] >= min_lift))
    return pairs.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def popular_items(baskets: pd.DataFrame) -> list[str]:
    """เมนูเรียงจากขายดี (อยู่ในบิลมากที่สุด) ใช้เป็นคำแนะนำสำรองและ Baseline"""
    return baskets[item_columns(baskets)].sum().sort_values(ascending=False).index.tolist()


# ---------------------------------------------------------------------------
# แนะนำเมนู
# ---------------------------------------------------------------------------
def recommend(items_in_bill: list[str], rules: pd.DataFrame, popular: list[str], k: int = TOP_K) -> pd.DataFrame:
    """
    แนะนำ k เมนูจากเมนูที่อยู่ในบิล
      1. ใช้กฎที่ผ่านเกณฑ์ซึ่งเงื่อนไข (A) อยู่ในบิล เรียงตาม Lift แล้วตาม Confidence
         (ถ้าหลายเมนูในบิลแนะนำเมนูเดียวกัน ใช้กฎที่ดีที่สุด)
      2. ไม่แนะนำเมนูที่อยู่ในบิลแล้ว
      3. ถ้ากฎให้คำแนะนำไม่ครบ k เมนู เติมด้วยเมนูขายดี (Fallback) และบอกว่าเป็นคำแนะนำสำรอง
    """
    in_bill = set(items_in_bill)
    # กฎที่อ่านจาก SQLite (reco_rules) เก็บเฉพาะกฎที่ผ่านเกณฑ์ จึงไม่มีคอลัมน์ passes -> ถือว่าผ่านทั้งหมด
    passed = rules["passes"] if "passes" in rules.columns else pd.Series(True, index=rules.index)
    ok = rules[passed & rules["antecedent"].isin(in_bill) & ~rules["consequent"].isin(in_bill)]
    best = (ok.sort_values(["lift", "confidence"], ascending=False)
              .drop_duplicates("consequent")
              .head(k))
    out = pd.DataFrame({"product_id": best["consequent"], "source": "rule", "because_of": best["antecedent"],
                        "confidence": best["confidence"], "lift": best["lift"]})
    if len(out) < k:
        extra = [p for p in popular if p not in in_bill and p not in set(out["product_id"])][:k - len(out)]
        out = pd.concat([out, pd.DataFrame({"product_id": extra, "source": "popular"})], ignore_index=True)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# วัดผล
# ---------------------------------------------------------------------------
def leave_one_out_cases(test: pd.DataFrame) -> list[tuple[list[str], str]]:
    """จากบิลที่มี 2 เมนูขึ้นไป: ซ่อนทีละเมนู -> (เมนูที่ระบบเห็น, เมนูที่ซ่อน)"""
    items = item_columns(test)
    cases = []
    for row in test[items].to_numpy():
        present = [items[i] for i in np.flatnonzero(row)]
        if len(present) < 2:
            continue
        for hidden in present:
            cases.append(([p for p in present if p != hidden], hidden))
    return cases


def evaluate_hit_rate(train: pd.DataFrame, test: pd.DataFrame, rules: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """
    Hit rate@1 และ Hit rate@k ของระบบแนะนำ เทียบกับ Baseline "แนะนำเมนูขายดีที่ยังไม่อยู่ในบิล"
    ใช้กฎและความนิยมจากข้อมูลฝึกเท่านั้น
    """
    popular = popular_items(train)
    cases = leave_one_out_cases(test)
    hit = {"model@1": 0, f"model@{k}": 0, "baseline@1": 0, f"baseline@{k}": 0}
    rule_based = 0
    for visible, hidden in cases:
        rec = recommend(visible, rules, popular, k)
        rule_based += int((rec["source"] == "rule").any())
        ids = rec["product_id"].tolist()
        hit["model@1"] += int(ids[:1] == [hidden])
        hit[f"model@{k}"] += int(hidden in ids)
        base = [p for p in popular if p not in visible][:k]
        hit["baseline@1"] += int(base[:1] == [hidden])
        hit[f"baseline@{k}"] += int(hidden in base)
    n = len(cases)
    return pd.DataFrame([
        {"method": "ระบบแนะนำ (Association rules)", "cases": n,
         "hit_rate@1": hit["model@1"] / n, f"hit_rate@{k}": hit[f"model@{k}"] / n,
         "cases_with_rule": rule_based / n},
        {"method": "Baseline (เมนูขายดี)", "cases": n,
         "hit_rate@1": hit["baseline@1"] / n, f"hit_rate@{k}": hit[f"baseline@{k}"] / n,
         "cases_with_rule": np.nan},
    ])


def rule_stability(rules: pd.DataFrame, test: pd.DataFrame, baskets: pd.DataFrame) -> pd.DataFrame:
    """
    กฎเชื่อถือได้ไหม
      ข้ามเวลา: Confidence ในข้อมูลฝึก เทียบกับข้อมูลทดสอบ (มิ.ย.)
      ข้ามสาขา: Lift ของกฎเดียวกันในแต่ละสาขา (ข้อมูลทั้งหมด)
    """
    passed = rules[rules["passes"]][["antecedent", "consequent", "confidence", "lift"]].copy()
    test_rules = mine_rules(test).set_index(["antecedent", "consequent"])
    key = list(zip(passed["antecedent"], passed["consequent"]))
    passed["confidence_test"] = [test_rules["confidence"].get(k, np.nan) for k in key]
    for branch, g in baskets.groupby("branch"):
        br = mine_rules(g).set_index(["antecedent", "consequent"])
        passed[f"lift_{branch}"] = [br["lift"].get(k, np.nan) for k in key]
    return passed.reset_index(drop=True)


def threshold_sensitivity(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """ถ้าเปลี่ยนเกณฑ์กรองกฎ จำนวนกฎและความแม่นเปลี่ยนอย่างไร"""
    rows = []
    for s, c, l in [(0.005, 0.10, 1.0), (0.01, 0.20, 1.2), (0.02, 0.30, 1.5), (0.03, 0.40, 2.0), (0.05, 0.40, 2.0)]:
        r = mine_rules(train, s, c, l)
        ev = evaluate_hit_rate(train, test, r).iloc[0]
        rows.append({"min_support": s, "min_confidence": c, "min_lift": l, "rules": int(r["passes"].sum()),
                     "hit_rate@1": ev["hit_rate@1"], f"hit_rate@{TOP_K}": ev[f"hit_rate@{TOP_K}"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# บันทึกกฎลง SQLite
# ---------------------------------------------------------------------------
def save_rules(rules: pd.DataFrame, products: pd.DataFrame, db_path: Path = load.DB_PATH) -> pd.DataFrame:
    """บันทึกกฎที่ผ่านเกณฑ์ลงตาราง reco_rules (Full refresh ใน Transaction เดียว) ให้หน้าเว็บอ่าน"""
    names = products.set_index("product_id")["name_th"]
    out = rules[rules["passes"]].drop(columns="passes").copy()
    out.insert(1, "antecedent_name", out["antecedent"].map(names))
    out.insert(3, "consequent_name", out["consequent"].map(names))
    with closing(load.connect(db_path)) as con:
        with con:
            con.execute("DROP TABLE IF EXISTS reco_rules")
            con.execute("""
                CREATE TABLE reco_rules (
                    antecedent      TEXT NOT NULL REFERENCES dim_product (product_id),
                    antecedent_name TEXT NOT NULL,
                    consequent      TEXT NOT NULL REFERENCES dim_product (product_id),
                    consequent_name TEXT NOT NULL,
                    count_ab        INTEGER NOT NULL,
                    count_a         INTEGER NOT NULL,
                    support         REAL NOT NULL CHECK (support BETWEEN 0 AND 1),
                    confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    lift            REAL NOT NULL CHECK (lift > 0),
                    PRIMARY KEY (antecedent, consequent)
                )""")
            con.executemany(f"INSERT INTO reco_rules ({', '.join(out.columns)}) VALUES ({', '.join('?' * out.shape[1])})",
                            [tuple(load._to_sql_value(v) for v in r) for r in out.itertuples(index=False, name=None)])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# รันทั้งโมดูล
# ---------------------------------------------------------------------------
def run_recommend(db_path: Path = load.DB_PATH, save: bool = True) -> dict:
    baskets = load_baskets(db_path)
    products = load.read_table("dim_product", db_path)
    train, test = split_by_time(baskets)
    pairs = mine_rules(train)
    popular = popular_items(train)
    names = products.set_index("product_id")["name_th"]
    examples = pd.concat([recommend([p], pairs, popular).assign(customer_ordered=p) for p in products["product_id"]],
                         ignore_index=True)
    examples["customer_ordered"] = examples["customer_ordered"].map(names)
    examples["recommend"] = examples["product_id"].map(names)
    examples["because_of"] = examples["because_of"].map(names)
    result = {
        "train_bills": len(train), "test_bills": len(test),
        "pairs": pairs, "rules": pairs[pairs["passes"]].reset_index(drop=True), "popular": popular,
        "evaluation": evaluate_hit_rate(train, test, pairs),
        "stability": rule_stability(pairs, test, baskets),
        "sensitivity": threshold_sensitivity(train, test),
        "examples": examples[["customer_ordered", "recommend", "source", "because_of", "confidence", "lift"]],
    }
    if save:
        result["saved_rules"] = save_rules(pairs, products, db_path)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res = run_recommend()
    names = load.read_table("dim_product").set_index("product_id")["name_th"]
    r = res["rules"].assign(A=lambda d: d["antecedent"].map(names), B=lambda d: d["consequent"].map(names))
    print(f"ข้อมูลฝึก {res['train_bills']:,} บิล / ทดสอบ {res['test_bills']:,} บิล")
    print(f"\n=== กฎที่ผ่านเกณฑ์ ({len(r)} กฎ) ===")
    print(r[["A", "B", "count_ab", "support", "confidence", "lift"]].round(3).to_string(index=False))
    print("\n=== วัดผล (Leave-one-out กับข้อมูล มิ.ย.) ===")
    print(res["evaluation"].round(3).to_string(index=False))
    print("\n=== ความมั่นคงของกฎ (ข้ามเวลา / ข้ามสาขา) ===")
    st = res["stability"].assign(A=lambda d: d["antecedent"].map(names), B=lambda d: d["consequent"].map(names))
    print(st.drop(columns=["antecedent", "consequent"]).round(3).to_string(index=False))
    print("\n=== ถ้าเปลี่ยนเกณฑ์ ===")
    print(res["sensitivity"].round(3).to_string(index=False))
    print("\n=== ตัวอย่างคำแนะนำ ===")
    print(res["examples"].round(3).to_string(index=False))
