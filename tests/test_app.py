"""
ทดสอบโมดูล 9 (หน้าเว็บ) ด้วย streamlit.testing (AppTest) ของ Streamlit เอง
รัน pipeline ทั้งสายลงฐานข้อมูลชั่วคราว แล้วเปิดทุกหน้าให้แน่ใจว่าไม่มี error (ทดสอบการทำงานต่อกันทั้งสาย)
"""

import importlib
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "raw", "extract", "quality", "clean", "transform_load", "dashboard", "recommend", "forecast"]
TIMEOUT = 300


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """ชี้ฐานข้อมูลและโฟลเดอร์อัปโหลดไปที่โฟลเดอร์ชั่วคราว แล้วรัน pipeline ทั้งสายเหมือนกดปุ่มบนหน้าเว็บ"""
    tmp = tmp_path_factory.mktemp("app")
    os.environ["DOIBREW_DB"] = str(tmp / "app.db")
    os.environ["DOIBREW_UPLOAD"] = str(tmp / "upload")
    from webapp import data as module
    module = importlib.reload(module)
    module.run_full_pipeline(module.DEFAULT_RAW, module.DEFAULT_DB)
    yield module
    module.clear_caches()
    for key in ["DOIBREW_DB", "DOIBREW_UPLOAD"]:
        os.environ.pop(key, None)


def page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "webapp" / "pages" / f"{name}.py"), default_timeout=TIMEOUT)


# ---- เปิดทุกหน้าต้องไม่มี error ----
@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_without_error(data, name):
    at = page(name).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", PAGES)
def test_tables_render_without_type_conversion_errors(data, name):
    """
    ตารางที่ปนตัวเลขกับข้อความในคอลัมน์เดียว ทำให้ตัวแปลงตาราง (Arrow) ของ Streamlit ล้มเหลว
    หน้าเว็บยังแสดงได้แต่มี error ใน log จึงดักข้อความใน log แทนการดู exception
    """
    import io
    import logging
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("streamlit.dataframe_util")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        at = page(name).run()
        for b in at.button:            # กดทุกปุ่มในหน้า (เช่น สาธิต Incremental loading)
            b.click().run()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert "Conversion failed" not in buf.getvalue()


def test_main_app_renders(data):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("รัน pipeline ใหม่" in b.label for b in at.sidebar.button)


# ---- การใช้งานบนหน้าเว็บ ----
def test_recommendation_interaction(data):
    at = page("recommend").run()
    at.multiselect[0].set_value(["P01"]).run()      # เอสเปรสโซ่
    assert not at.exception
    assert any("แซนด์วิชแฮมชีส" in s.value for s in at.success)


@pytest.mark.parametrize("n_items", [0, 1, 13, 14, 15])
def test_recommendation_any_number_of_items(data, n_items):
    """เลือกเมนูกี่เมนูก็ได้ รวมถึงเลือกครบทุกเมนู (ไม่มีเมนูเหลือให้แนะนำ) ต้องไม่ error"""
    at = page("recommend").run()
    at.multiselect[0].set_value([f"P{i:02d}" for i in range(1, n_items + 1)]).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("months", [(1, 1), (6, 6), (1, 6)])
def test_dashboard_month_ranges(data, months):
    at = page("dashboard").run()
    at.select_slider[0].set_range(*months).run()
    assert not at.exception


def test_dashboard_no_branch_shows_warning(data):
    at = page("dashboard").run()
    at.multiselect[0].set_value([]).run()
    assert not at.exception and at.warning


@pytest.mark.parametrize("branch", ["nimman", "cmu", "thaphae"])
def test_forecast_each_branch(data, branch):
    at = page("forecast").run()
    at.radio[0].set_value(branch).run()
    assert not at.exception


def test_sql_examples_all_run(data):
    at = page("transform_load").run()
    for option in at.selectbox[0].options:
        at.selectbox[0].set_value(option).run()
        [b for b in at.button if b.label == "รัน SQL"][0].click().run()
        assert not at.exception and not at.error, option


def test_dashboard_filter_single_branch(data):
    at = page("dashboard").run()
    at.multiselect[0].set_value(["cmu"]).run()
    assert not at.exception


def test_sql_box_rejects_write(data):
    at = page("transform_load").run()
    at.text_area[0].input("DELETE FROM fact_sales").run()
    [b for b in at.button if b.label == "รัน SQL"][0].click().run()
    assert not at.exception
    assert any("รันไม่ได้" in e.value for e in at.error)
    assert data.sql("SELECT COUNT(*) n FROM fact_sales").iloc[0]["n"] > 0   # ข้อมูลยังอยู่ครบ


# ---- ช่อง SQL แบบอ่านอย่างเดียว ----
@pytest.mark.parametrize("bad", [
    "DELETE FROM fact_sales", "DROP TABLE dim_date", "SELECT 1; DELETE FROM fact_sales",
    "UPDATE dim_member SET gender = 'M'", "PRAGMA table_info(fact_sales)", "",
])
def test_safe_select_rejects(data, bad):
    with pytest.raises(ValueError):
        data.safe_select(bad)


def test_safe_select_allows_read(data):
    df = data.safe_select("SELECT branch_id, SUM(line_total) revenue FROM fact_sales GROUP BY branch_id;")
    assert len(df) == 3
    df = data.safe_select("WITH x AS (SELECT * FROM dim_date) SELECT COUNT(*) n FROM x")
    assert df["n"].iloc[0] == 181


def test_safe_select_is_really_read_only(data):
    """ถึงผ่านการตรวจข้อความ ฐานข้อมูลก็ต้องเปิดแบบอ่านอย่างเดียว"""
    import sqlite3
    path = data.DEFAULT_DB.resolve()
    with pytest.raises(sqlite3.OperationalError):
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            con.execute("DELETE FROM dim_branch")
        finally:
            con.close()


# ---- อัปโหลดไฟล์ ----
def _raw_bytes(data):
    return {k: (data.DEFAULT_RAW / name).read_bytes() for k, (name, _) in data.UPLOAD_SPEC.items()}


def test_validate_upload_accepts_sample_files(data):
    for kind, content in _raw_bytes(data).items():
        assert data.validate_upload(kind, content) == [], kind


def test_validate_upload_rejects_bad_files(data):
    assert data.validate_upload("members", "member_id,full_name\nM1,x\n".encode("utf-8"))   # คอลัมน์ไม่ครบ
    assert data.validate_upload("thaphae", b"not json")                                     # อ่านไม่ได้
    assert data.validate_upload("cmu", b"not excel")
    assert data.validate_upload("thaphae", b'{"receipts": []}')                              # ไม่มีบิล


def test_upload_end_to_end(data):
    """อัปโหลดไฟล์ -> รัน pipeline ลงฐานข้อมูลแยก -> ข้อมูลตัวอย่างไม่ถูกทับ"""
    before = data.sql("SELECT COUNT(*) n FROM fact_sales").iloc[0]["n"]
    raw = data.save_uploads(_raw_bytes(data))
    data.run_full_pipeline(raw, data.UPLOAD_DB)
    assert data.UPLOAD_DB.exists()
    assert data.UPLOAD_DB != data.DEFAULT_DB
    assert data.sql("SELECT COUNT(*) n FROM fact_sales").iloc[0]["n"] == before


# ---- กฎของโปรเจกต์ ----
def test_webapp_does_not_read_answer_key():
    files = [ROOT / "app.py", *(ROOT / "webapp").rglob("*.py")]
    for f in files:
        assert "generator_log" not in f.read_text(encoding="utf-8"), f.name


def test_upload_failure_cleans_up(data, monkeypatch):
    """ถ้ารัน pipeline ล้มเหลว ต้องไม่เหลือฐานข้อมูลที่สร้างไม่เสร็จให้เลือกบนหน้าเว็บ"""
    def boom(*args, **kwargs):
        raise RuntimeError("จำลองความล้มเหลว")
    monkeypatch.setattr(data, "run_full_pipeline", boom)
    problems = data.run_upload(_raw_bytes(data))
    assert problems and "ไม่สำเร็จ" in problems[0]
    assert not data.UPLOAD_ROOT.exists()


def test_upload_requires_all_files(data):
    files = _raw_bytes(data)
    files.pop("members")
    assert data.run_upload(files)


def test_home_shows_integer_counts(data):
    at = page("home").run()
    bills = [m.value for m in at.metric if m.label == "จำนวนบิล"][0]
    assert "." not in bills
