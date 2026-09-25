"""
วัดความแม่นของโมดูล 7 (Recommendation) เทียบกับคู่เมนูที่ใส่ไว้ตอนสร้างข้อมูล (data_generator/generate.py)

วิธีรัน:  py -3.13 -m evaluation.evaluate_recommend
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline import recommend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_generator"))
import generate  # noqa: E402


def planted_pairs() -> set[tuple[str, str]]:
    """คู่เมนูที่ใส่ไว้ (ทั้งสองทิศทาง เพราะการซื้อคู่กันเกิดได้ทั้ง A -> B และ B -> A)"""
    return {(a, b) for a, b in generate.PAIRS.items()} | {(b, a) for a, b in generate.PAIRS.items()}


def evaluate(res: dict | None = None) -> pd.DataFrame:
    res = res or recommend.run_recommend(save=False)
    found = set(zip(res["rules"]["antecedent"], res["rules"]["consequent"]))
    truth = planted_pairs()
    return pd.DataFrame([
        {"metric": "กฎที่ควรเจอ (คู่ที่ใส่ไว้ x 2 ทิศทาง)", "value": len(truth)},
        {"metric": "กฎที่ระบบเจอ (ผ่านเกณฑ์)", "value": len(found)},
        {"metric": "เจอถูก", "value": len(found & truth)},
        {"metric": "กฎแปลกปลอม (ไม่ได้ใส่ไว้แต่ผ่านเกณฑ์)", "value": len(found - truth)},
        {"metric": "คู่ที่ใส่ไว้แต่หาไม่เจอ", "value": len(truth - found)},
    ])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(evaluate().to_string(index=False))
