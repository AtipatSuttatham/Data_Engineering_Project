"""
รัน pipeline ทั้งสายด้วยคำสั่งเดียว (โมดูล 10)

ETL เป็นกระบวนการอัตโนมัติ (Ch1: "ETL is an automated process") คำสั่งนี้รันโมดูล 2 -> 8 ตามลำดับ
แล้วพิมพ์สรุปผลของแต่ละขั้น หน้าเว็บ (ปุ่ม "รัน pipeline ใหม่") ก็เรียกใช้ฟังก์ชัน run() ตัวเดียวกันนี้

วิธีรัน:
    py -3.13 -m pipeline.run_all               รันทุกขั้นจากข้อมูลดิบที่มีอยู่
    py -3.13 -m pipeline.run_all --regenerate  สร้างข้อมูลดิบใหม่ (โมดูล 1) ก่อน แล้วรันทุกขั้น
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pipeline import clean, extract, forecast, load, recommend, transform

ROOT = Path(__file__).resolve().parents[1]

STEPS = ["Extract (โมดูล 2)", "Quality + Cleaning (โมดูล 3-4)", "Transform (โมดูล 5)",
         "Load เข้า SQLite (โมดูล 6)", "Recommendation (โมดูล 7)", "Forecast (โมดูล 8)"]


def regenerate_raw_data() -> None:
    """สร้างข้อมูลดิบใหม่ด้วยโมดูล 1 (seed คงที่ ได้ข้อมูลเหมือนเดิมทุกครั้ง)"""
    sys.path.insert(0, str(ROOT / "data_generator"))
    import generate
    generate.main(out_root=ROOT, verbose=False)


def run(raw_dir: Path = extract.RAW_DIR, db_path: Path = load.DB_PATH, on_step=None) -> dict:
    """
    รันโมดูล 2 -> 8 ตามลำดับ on_step(i, ชื่อขั้น) ใช้แสดงความคืบหน้า
    คืนเฉพาะตัวเลขสรุป (ไม่เก็บตารางใหญ่ไว้ เพื่อประหยัดหน่วยความจำ)
    """
    step = on_step or (lambda i, name: None)
    started = time.perf_counter()

    step(0, STEPS[0])
    ext = extract.run_extract(raw_dir=raw_dir, save=False)
    step(1, STEPS[1])
    cleaned = clean.run_clean(save=False, ext=ext)
    step(2, STEPS[2])
    tables = transform.run_transform(save=False, cleaned=cleaned)
    step(3, STEPS[3])
    loaded = load.run_load(transformed=tables, db_path=db_path)
    step(4, STEPS[4])
    reco = recommend.run_recommend(db_path=db_path, save=True)
    step(5, STEPS[5])
    fc = forecast.run_forecast(db_path=db_path, save=True)

    total = ext["report"].set_index("branch").loc["total"]
    model_eval = reco["evaluation"].iloc[0]
    wf = fc["walk_forward"]
    return {
        "rows_extracted": int(total["rows_out"]),
        "quality_before": cleaned["quality_before"].overall["score_avg"],
        "quality_after": cleaned["quality_after"].overall["score_avg"],
        "rows_clean": len(cleaned["sales"]),
        "rows_removed": len(cleaned["removed"]),
        "members": len(cleaned["members"]),
        "revenue": float(tables["fact_sales"]["line_total"].sum()),
        "load_verification_passed": bool(loaded["verification"]["passed"].all()),
        "reco_rules": len(reco["rules"]),
        "reco_hit_rate_1": float(model_eval["hit_rate@1"]),
        "reco_baseline_hit_rate_1": float(reco["evaluation"].iloc[1]["hit_rate@1"]),
        "forecast_model": fc["chosen"],
        "forecast_months_beating_baselines": int(wf["beats_both_baselines"].sum()),
        "forecast_months_tested": len(wf),
        "db_path": str(db_path),
        "seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="รัน pipeline ของ Doi Brew ทั้งสาย (โมดูล 2 -> 8)")
    parser.add_argument("--regenerate", action="store_true", help="สร้างข้อมูลดิบใหม่ด้วยโมดูล 1 ก่อน")
    args = parser.parse_args(argv)
    if args.regenerate:
        print("[0/6] สร้างข้อมูลดิบใหม่ (โมดูล 1)")
        regenerate_raw_data()
    s = run(on_step=lambda i, name: print(f"[{i + 1}/{len(STEPS)}] {name}", flush=True))
    print("\n=== สรุป ===")
    print(f"อ่านข้อมูลเข้า          : {s['rows_extracted']:,} แถว")
    print(f"คะแนนคุณภาพข้อมูล      : {s['quality_before']:.1%} -> {s['quality_after']:.1%}")
    print(f"หลังทำความสะอาด        : {s['rows_clean']:,} แถว (ตัดออก {s['rows_removed']:,}) / สมาชิก {s['members']:,} คน")
    print(f"ยอดขายรวม              : {s['revenue']:,.0f} บาท")
    print(f"Load verification      : {'ผ่านทุกข้อ' if s['load_verification_passed'] else 'ไม่ผ่าน'}")
    print(f"Recommendation         : {s['reco_rules']} กฎ / ทายอันดับ 1 ถูก {s['reco_hit_rate_1']:.1%} "
          f"(Baseline {s['reco_baseline_hit_rate_1']:.1%})")
    print(f"Forecast               : โมเดล {s['forecast_model']} ชนะ Baseline "
          f"{s['forecast_months_beating_baselines']} จาก {s['forecast_months_tested']} เดือน")
    print(f"ฐานข้อมูล               : {s['db_path']}")
    print(f"ใช้เวลา                 : {s['seconds']:.0f} วินาที")
    return s


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # พิมพ์ภาษาไทยได้แม้ส่งผลลัพธ์ไปไฟล์หรือ pipe บน Windows
    main()
