"""
ขั้นตอนนำเข้าไฟล์ของหน้า "อัปโหลดไฟล์" (บันทึกไฟล์ -> ตรวจคุณภาพและทำความสะอาด -> โหลด)

ไม่มีตรรกะ DE ใหม่ ทุกขั้นเรียกฟังก์ชันใน pipeline/ ตัวเดียวกับที่ pipeline.run_all ใช้
    อ่าน + Schema mapping + Translation mapping  -> pipeline.extract   (Ch1, Ch2, Ch5)
    วัดคุณภาพ + ทำความสะอาด                      -> pipeline.clean / quality (Ch3, Ch4)
    แปลง + โหลด + ตรวจหลังโหลด                   -> pipeline.transform / load (Ch1, Ch2)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline import clean, extract, forecast, load, recommend, transform
from webapp import data


def stage(files: dict[str, bytes], folder: Path) -> Path:
    """บันทึกไฟล์ด้วยชื่อมาตรฐานในโฟลเดอร์ชั่วคราว + Master data (สินค้า, วันหยุด) จากข้อมูลตัวอย่าง"""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for kind, content in files.items():
        (folder / data.UPLOAD_SPEC[kind][0]).write_bytes(content)
    for master in ["products.csv", "holidays.csv"]:
        shutil.copy(data.DEFAULT_RAW / master, folder / master)
    return folder


def run_checks(folder: Path) -> dict:
    """Extract + วัดคุณภาพ + ทำความสะอาด (ยังไม่แตะฐานข้อมูล) คืนผลของ pipeline.clean.run_clean"""
    ext = extract.run_extract(raw_dir=folder, save=False)
    cleaned = clean.run_clean(save=False, ext=ext)
    cleaned["extract_report"] = ext["report"]
    return cleaned


def load_upload(files: dict[str, bytes], cleaned: dict) -> dict:
    """
    โหลดข้อมูลที่ตรวจแล้วเข้าฐานข้อมูลของชุด "ข้อมูลที่อัปโหลด" (แยกจากข้อมูลตัวอย่าง) แบบ Full refresh
    แล้วคำนวณโมเดลใหม่ คืน dict: verification (ตาราง Load verification ของ Ch1), load_log
    ถ้าล้มเหลวกลางทาง ลบไฟล์และฐานข้อมูลที่ค้าง ไม่ให้เหลือชุดข้อมูลที่ใช้ไม่ได้ (เหมือน webapp.data.run_upload)
    """
    try:
        data.save_uploads(files)
        tables = transform.run_transform(save=False, cleaned=cleaned)
        loaded = load.run_load(transformed=tables, db_path=data.UPLOAD_DB)
        recommend.run_recommend(db_path=data.UPLOAD_DB, save=True)
        forecast.run_forecast(db_path=data.UPLOAD_DB, save=True)
    except Exception:
        shutil.rmtree(data.UPLOAD_ROOT, ignore_errors=True)
        raise
    finally:
        data.clear_caches()
    return {"verification": loaded["verification"], "load_log": loaded["load_log"]}
