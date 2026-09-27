"""
ทดสอบโมดูล 10: คำสั่งรันทุกขั้น (pipeline/run_all.py) และสคริปต์สร้างไฟล์ส่งงาน (scripts/make_submission.py)
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from pipeline import load, run_all

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_submission  # noqa: E402


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    return run_all.run(db_path=tmp_path_factory.mktemp("runall") / "all.db")


def test_run_all_summary(summary):
    assert summary["quality_after"] > summary["quality_before"]
    assert summary["load_verification_passed"]
    assert summary["reco_hit_rate_1"] > summary["reco_baseline_hit_rate_1"]
    assert summary["rows_clean"] + summary["rows_removed"] == summary["rows_extracted"]


def test_run_all_creates_every_table(summary):
    db = Path(summary["db_path"])
    tables = set(load.query("SELECT name FROM sqlite_master WHERE type = 'table'", db)["name"])
    expected = set(load.LOAD_ORDER) | set(load.DERIVED_TABLES) | {"etl_load_log"}
    assert expected <= tables


def test_run_all_reports_progress(tmp_path):
    seen = []
    run_all.run(db_path=tmp_path / "p.db", on_step=lambda i, name: seen.append(i))
    assert seen == list(range(len(run_all.STEPS)))


def test_webapp_uses_same_runner():
    """หน้าเว็บต้องเรียก run_all ตัวเดียวกัน (มีโค้ดชุดเดียว ไม่ซ้ำสองที่)"""
    text = (ROOT / "webapp" / "data.py").read_text(encoding="utf-8")
    assert "run_all.run(" in text


def test_cli_prints_thai_through_pipe():
    """เครื่อง Windows ทั่วไปใช้ cp1252 เมื่อส่งผลลัพธ์ไป pipe หรือไฟล์ ภาษาไทยต้องไม่ทำให้โปรแกรมล้ม"""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONIOENCODING"] = "cp1252"
    out = subprocess.run([sys.executable, "-m", "pipeline.run_all", "--help"], cwd=ROOT, env=env,
                         capture_output=True)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert "สร้างข้อมูลดิบใหม่".encode("utf-8") in out.stdout


def test_requirements_is_ascii():
    """pip รุ่นเก่า (22.3 ที่มากับ Python 3.11.0) อ่านไฟล์นี้ด้วย cp1252 ถ้ามีภาษาไทยจะติดตั้งไม่ได้"""
    (ROOT / "requirements.txt").read_bytes().decode("ascii")


def test_every_cli_sets_utf8_output():
    files = [*(ROOT / "pipeline").glob("*.py"), *(ROOT / "evaluation").glob("evaluate_*.py"),
             ROOT / "data_generator" / "generate.py", ROOT / "scripts" / "make_submission.py"]
    for p in files:
        text = p.read_text(encoding="utf-8")
        if '__name__ == "__main__"' in text:
            assert 'sys.stdout.reconfigure(encoding="utf-8")' in text, p.name


# สร้าง zip ต้องใช้ git แต่โฟลเดอร์ที่แตกจาก zip ส่งงานไม่มี .git จึงข้ามข้อนี้ (reason เป็นอังกฤษ เพราะ pytest แสดงภาษาไทยเป็นรหัส \u)
@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git repository (the submission zip has no .git)")
def test_submission_zip(tmp_path):
    """zip ต้องมีไฟล์สำคัญ + ฐานข้อมูล และไม่มีเนื้อหาวิชา / ไฟล์ชั่วคราว"""
    committed = ["app.py", "pipeline/extract.py", "data/raw/nimman_sales.csv", "CLAUDE.md"]
    path = make_submission.build("ทดสอบกลุ่ม", out_dir=tmp_path, db_path=tmp_path / "sub.db", allow_dirty=True,
                                 required=committed)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert "ทดสอบกลุ่ม/data/processed/doibrew.db" in names
    assert not any("Lecture/" in n or n.endswith(".pdf") or "__pycache__" in n for n in names)


def test_submission_refuses_uncommitted_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(make_submission, "uncommitted_changes", lambda: [" M app.py"])
    with pytest.raises(RuntimeError):
        make_submission.build("x", out_dir=tmp_path, include_db=False)


def test_verify_detects_forbidden_files(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("g/Lecture/ch-1.pdf", "x")
        z.writestr("g/app.py", "x")
    problems = make_submission.verify(bad, "g", include_db=False, required=["app.py", "README.md"])
    assert any("README.md" in p for p in problems)
    assert any("ต้องห้าม" in p for p in problems)
