"""
วัดความแม่นของโมดูล 4 (ทำความสะอาด) เทียบกับไฟล์เฉลย
ใช้ตัวเลขนี้ในเล่มรายงาน เช่น "เติมขนาดแก้วถูก 100%" หรือ "ทายอายุคลาดเฉลี่ย 4 ปี"

วิธีรัน:  py -3.13 -m evaluation.evaluate_cleaning
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import clean

LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "generator_log"


def load_truth() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sales = pd.read_csv(LOG_DIR / "sales_truth.csv", encoding="utf-8-sig", parse_dates=["sale_dt"])
    sales["sale_dt"] = sales["sale_dt"].dt.tz_localize("Asia/Bangkok")
    members = pd.read_csv(LOG_DIR / "members_truth.csv", encoding="utf-8-sig")
    issues = json.loads((LOG_DIR / "injected_issues.json").read_text(encoding="utf-8"))
    return sales, members, issues


def issue_lines(issues: dict, truth: pd.DataFrame, issue_id: str) -> set[tuple]:
    """(receipt_id, product_id) ของแถวที่ใส่ปัญหา issue_id ไว้"""
    idx = truth.set_index(["receipt_id", "line_no"])["product_id"]
    return {(r["receipt_id"], idx[(r["receipt_id"], r["line_no"])])
            for r in issues["issues"] if r["issue_id"] == issue_id}


def evaluate(res: dict | None = None) -> dict:
    res = res or clean.run_clean(save=False)
    truth, mtruth, issues = load_truth()
    s = res["sales"]
    out: dict = {}

    # 1) แถวที่ตัดออก ตรงกับที่ควรตัดไหม (ตัดได้แค่ B10 จำนวน <= 0 และ C1 จำนวนพิมพ์ผิด)
    key_clean = set(zip(s["receipt_id"], s["product_id"]))
    key_truth = set(zip(truth["receipt_id"], truth["product_id"]))
    should_drop = issue_lines(issues, truth, "B10") | issue_lines(issues, truth, "C1")
    dropped = key_truth - key_clean
    out["rows"] = pd.DataFrame([
        {"metric": "แถวในข้อมูลจริง (เฉลย)", "value": len(key_truth)},
        {"metric": "แถวหลังทำความสะอาด", "value": len(s)},
        {"metric": "แถวที่ควรตัด (B10 + C1)", "value": len(should_drop)},
        {"metric": "แถวที่ตัดจริง", "value": len(dropped)},
        {"metric": "ตัดถูก", "value": len(dropped & should_drop)},
        {"metric": "ตัดผิด (ไม่ควรตัดแต่ตัด)", "value": len(dropped - should_drop)},
        {"metric": "แถวแปลกปลอม (ไม่มีในเฉลย)", "value": len(key_clean - key_truth)},
    ])

    # 2) ค่าที่แก้ / เติม ถูกต้องไหม (เทียบรายแถว)
    m = s.merge(truth, on=["receipt_id", "product_id"], suffixes=("", "_t"))
    checks = [
        ("product_synonym", "product_id", "product_id"),   # ชื่อเมนูที่แก้ -> ต้องได้สินค้าตรงเฉลย
        ("size_from_price", "size", "size_t"),
        ("qty_from_total", "qty", "qty_t"),
        ("price_from_list", "unit_price", "unit_price_t"),
        ("price_fixed_from_list", "unit_price", "unit_price_t"),
        ("member_id_fixed", "member_id", "member_id_t"),
        ("member_id_merged", "member_id", "member_id_t"),
    ]
    rows = []
    for flag, col, tcol in checks:
        g = m[m["clean_flags"].str.contains(flag)]
        if flag == "product_synonym":
            correct = len(g)  # merge สำเร็จด้วย product_id = ตรงเฉลยแล้ว
        else:
            correct = int((g[col].astype(str) == g[tcol].astype(str)).sum())
        rows.append({"fix": flag, "rows": len(g), "correct": correct,
                     "accuracy": correct / len(g) if len(g) else np.nan})
    unknown = m[m["clean_flags"].str.contains("member_id_unknown")]
    rows.append({"fix": "member_id_unknown (ตั้งเป็นว่าง)", "rows": len(unknown), "correct": np.nan,
                 "accuracy": np.nan})
    out["fixes"] = pd.DataFrame(rows)

    # ทุกคอลัมน์ของทุกแถว (ยกเว้นเวลาที่ติดธง time_suspect) ต้องตรงเฉลย
    cols = [("sale_datetime", "sale_dt"), ("size", "size_t"), ("qty", "qty_t"),
            ("unit_price", "unit_price_t"), ("payment", "payment_t")]
    mism = {}
    for c, t in cols:
        a, b = m[c], m[t]
        if c == "sale_datetime":
            diff = (a != b) & ~m["time_suspect"]
        else:
            diff = a.fillna("-").astype(str) != b.fillna("-").astype(str)
        mism[c] = int(diff.sum())
    mem_diff = (m["member_id"].fillna("-") != m["member_id_t"].fillna("-"))
    mism["member_id (ไม่นับที่ตั้งเป็นว่าง)"] = int((mem_diff & ~m["clean_flags"].str.contains("member_id_unknown")).sum())
    out["mismatch"] = pd.DataFrame([{"column": k, "rows_not_matching_truth": v} for k, v in mism.items()])

    # 3) Outlier: แต่ละวิธีจับข้อผิดพลาด (C1) ได้กี่แถว และจับออเดอร์ใหญ่จริง (C3) ผิดกี่แถว
    staged = res["outlier_flags"].join(
        clean.extract.run_extract(save=False)["sales"][["receipt_id", "product_id"]], how="left")
    c1 = issue_lines(issues, truth, "C1")
    c3_ids = {r["receipt_id"] for r in issues["issues"] if r["issue_id"] == "C3"}
    is_c1 = [(r, p) in c1 for r, p in zip(staged["receipt_id"], staged["product_id"])]
    is_c3 = staged["receipt_id"].isin(c3_ids)
    orows = []
    for method in ["zscore", "iqr", "dbscan", "isolation_forest"]:
        f = staged[method]
        orows.append({"method": method,
                      "C1_typos_caught": int((f & pd.Series(is_c1, index=staged.index)).sum()),
                      "C1_total": len(c1),
                      "C3_real_orders_flagged": int((f & is_c3).sum()),
                      "C3_total": len(c3_ids),
                      "normal_rows_flagged": int((f & ~pd.Series(is_c1, index=staged.index) & ~is_c3).sum())})
    final_c1_removed = len(c1 & dropped)
    orows.append({"method": "final (Z-score + > 50 แก้ว)", "C1_typos_caught": final_c1_removed,
                  "C1_total": len(c1),
                  "C3_real_orders_flagged": len(c3_ids - set(s["receipt_id"])),
                  "C3_total": len(c3_ids), "normal_rows_flagged": 0})
    out["outliers"] = pd.DataFrame(orows)

    # 4) สมาชิก: ปีเกิดที่แก้ (พ.ศ.) / ปีเกิดที่เติม / เพศ
    mem = res["members"].merge(mtruth, on="member_id", suffixes=("", "_t"))
    imp = mem[mem["birth_year_imputed"]]
    real = mem[~mem["birth_year_imputed"]]
    median_guess = real["birth_year"].astype(float).median()
    out["members"] = pd.DataFrame([
        {"metric": "สมาชิกหลังรวมคนซ้ำ", "value": len(res["members"])},
        {"metric": "ปีเกิดที่ไม่ได้เติม ตรงเฉลย", "value": f"{int((real['birth_year'] == real['birth_year_t']).sum())} / {len(real)}"},
        {"metric": "ปีเกิดที่เติม: คลาดเฉลี่ย (ปี) วิธีที่เลือก",
         "value": round(float((imp["birth_year"].astype(float) - imp["birth_year_t"]).abs().mean()), 2)},
        {"metric": "ปีเกิดที่เติม: คลาดเฉลี่ย (ปี) ถ้าใช้ Median",
         "value": round(float((median_guess - imp["birth_year_t"]).abs().mean()), 2)},
        {"metric": "เพศที่ไม่ใช่ U ตรงเฉลย",
         "value": f"{int((mem.loc[mem['gender'] != 'U', 'gender'] == mem.loc[mem['gender'] != 'U', 'gender_t']).sum())}"
                  f" / {int((mem['gender'] != 'U').sum())}"},
        {"metric": "ถ้าใช้ Mode (F) กับเพศที่ว่าง จะถูก",
         "value": f"{int((mem.loc[mem['gender'] == 'U', 'gender_t'] == 'F').sum())} / {int((mem['gender'] == 'U').sum())}"},
    ])
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    r = evaluate()
    for name, title in [("rows", "แถวที่ตัดออก"), ("fixes", "ความแม่นของการแก้ / เติมค่า"),
                        ("mismatch", "ค่าที่ไม่ตรงเฉลยหลังทำความสะอาด"), ("outliers", "เทียบวิธีตรวจ outlier กับเฉลย"),
                        ("members", "สมาชิก")]:
        print(f"=== {title} ===")
        print(r[name].to_string(index=False))
        print()
