"""
สร้างไฟล์ส่งงาน "ชื่อกลุ่ม.zip" (โมดูล 10)

สิ่งที่อยู่ใน zip
    - ทุกไฟล์ที่ commit แล้ว (สร้างด้วย git archive) ได้แก่ โค้ด, ข้อมูลดิบ, ไฟล์เฉลย, เอกสาร, tests, README
    - ฐานข้อมูล data/processed/doibrew.db ที่สร้างใหม่ก่อนทำ zip (อาจารย์เปิดดูด้วย DB Browser ได้ทันที)
สิ่งที่ไม่อยู่ใน zip
    - เนื้อหาวิชา (Lecture, Lab, PDF), ไฟล์ชั่วคราว, ประวัติ git และไฟล์ที่ยังไม่ commit

วิธีรัน (หลัง commit ทุกอย่างแล้ว):
    py -3.13 scripts/make_submission.py --name "ชื่อกลุ่ม"
ผลลัพธ์: dist/ชื่อกลุ่ม.zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import load, run_all  # noqa: E402

DB_IN_ZIP = "data/processed/doibrew.db"
REQUIRED = ["README.md", "requirements.txt", "CLAUDE.md", "app.py", "pipeline/run_all.py", "pipeline/extract.py",
            "webapp/data.py", "data_generator/generate.py", "data/raw/nimman_sales.csv", "data/raw/cmu_sales.xlsx",
            "data/raw/thaphae_sales.json", "data/raw/members.csv", "data/raw/products.csv", "data/raw/holidays.csv",
            "docs/report_notes/00_overview.md", "tests/test_app.py", "evaluation/evaluate_cleaning.py"]
FORBIDDEN = ["Lecture/", "Lab/", ".pdf", "__pycache__", ".ipynb_checkpoints", ".git/", "dist/"]


def uncommitted_changes() -> list[str]:
    """ไฟล์ที่แก้แล้วยังไม่ commit (ไม่นับไฟล์ที่ .gitignore กันไว้)"""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def verify(zip_path: Path, name: str, include_db: bool, required: list[str] = REQUIRED) -> list[str]:
    """ตรวจ zip: มีไฟล์ที่จำเป็นครบ / ไม่มีไฟล์ต้องห้าม / มีฐานข้อมูล (ถ้าเลือกใส่)"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n[len(name) + 1:] for n in z.namelist()]
        bad = z.testzip()
    problems = [f"ไม่มีไฟล์ {r}" for r in required if r not in names]
    problems += [f"มีไฟล์ต้องห้าม {n}" for n in names for f in FORBIDDEN if f in n]
    if include_db and DB_IN_ZIP not in names:
        problems.append("ไม่มีฐานข้อมูล")
    if bad:
        problems.append(f"ไฟล์ใน zip เสีย: {bad}")
    return problems


def build(name: str, out_dir: Path = ROOT / "dist", include_db: bool = True, db_path: Path = load.DB_PATH,
          allow_dirty: bool = False, required: list[str] = REQUIRED) -> Path:
    dirty = uncommitted_changes()
    if dirty and not allow_dirty:
        raise RuntimeError("มีไฟล์ที่ยังไม่ commit กรุณา commit ก่อน (zip ต้องตรงกับที่อยู่บน GitHub):\n"
                           + "\n".join(dirty))
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"
    zip_path.unlink(missing_ok=True)
    # git archive: เอาเฉพาะไฟล์ที่ commit แล้ว ไม่มีไฟล์ที่ .gitignore กันไว้และไม่มีประวัติ git
    subprocess.run(["git", "archive", "--format=zip", f"--prefix={name}/", "-o", str(zip_path), "HEAD"],
                   cwd=ROOT, check=True)
    if include_db:
        run_all.run(db_path=db_path)   # สร้างฐานข้อมูลใหม่ให้ตรงกับโค้ดล่าสุด
        with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(db_path, f"{name}/{DB_IN_ZIP}")
    problems = verify(zip_path, name, include_db, required)
    if problems:
        raise RuntimeError("zip ไม่ผ่านการตรวจ:\n" + "\n".join(problems))
    return zip_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="สร้างไฟล์ส่งงาน ชื่อกลุ่ม.zip")
    parser.add_argument("--name", required=True, help="ชื่อกลุ่ม (ใช้เป็นชื่อไฟล์ zip ตามเกณฑ์การส่งงาน)")
    parser.add_argument("--no-db", action="store_true", help="ไม่ใส่ฐานข้อมูล (หน้าเว็บจะสร้างให้เองตอนเปิดครั้งแรก)")
    args = parser.parse_args(argv)
    path = build(args.name, include_db=not args.no_db)
    with zipfile.ZipFile(path) as z:
        n = len(z.namelist())
    print(f"สร้างแล้ว: {path} ({path.stat().st_size / 1e6:.1f} MB, {n} ไฟล์) ผ่านการตรวจครบ")
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # พิมพ์ภาษาไทยได้แม้ส่งผลลัพธ์ไปไฟล์หรือ pipe บน Windows
    main()
