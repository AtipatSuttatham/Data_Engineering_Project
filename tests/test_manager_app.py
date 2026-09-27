"""
ทดสอบหน้าเว็บ Doi Brew Manager (manager_app.py + manager/)
รัน pipeline ทั้งสายลงฐานข้อมูลชั่วคราว แล้วเปิดทุกหน้า (4 หน้า) ลองตัวเลือก สวิตช์รายละเอียด และลองนำเข้าไฟล์
"""

import importlib
import io
import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from manager import insights

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["upload", "overview", "dashboard", "forecast"]
TIMEOUT = 600


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """ชี้ฐานข้อมูลและโฟลเดอร์อัปโหลดไปที่โฟลเดอร์ชั่วคราว แล้วรัน pipeline ทั้งสาย"""
    tmp = tmp_path_factory.mktemp("manager")
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
    return AppTest.from_file(str(ROOT / "manager" / "pages" / f"{name}.py"), default_timeout=TIMEOUT)


def raw_files(module) -> dict[str, bytes]:
    return {k: (module.DEFAULT_RAW / v[0]).read_bytes() for k, v in module.UPLOAD_SPEC.items()}


# ---- เปิดทุกหน้าต้องไม่มี error ----
@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_without_error(data, name):
    at = page(name).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", PAGES)
def test_tables_render_without_type_conversion_errors(data, name):
    """ตารางที่ปนตัวเลขกับข้อความในคอลัมน์เดียว ทำให้ Arrow แปลงไม่ได้ (ดักข้อความใน log เหมือน test_app.py)"""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("streamlit.dataframe_util")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        page(name).run()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert "Conversion failed" not in buf.getvalue()


def test_main_app_renders(data):
    at = AppTest.from_file(str(ROOT / "manager_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ป้ายเทคนิค" in t.label for t in at.toggle)


def test_technique_tags_can_be_hidden(data):
    at = page("overview").run()
    assert any('class="tech"' in m.value for m in at.markdown)
    at.session_state["show_tech"] = False
    at.run()
    assert not any('class="tech"' in m.value for m in at.markdown)


# ---- ตัวเลือกบนหน้าเว็บ ----
@pytest.mark.parametrize("period", list(insights.PERIODS))
def test_overview_every_period(data, period):
    at = page("overview").run()
    at.selectbox(key="ov_period").set_value(period).run()
    assert not at.exception
    revenue = at.metric[0].value
    assert revenue.startswith("฿")
    if insights.PERIODS[period] is None:          # เลือกทั้งหมด: ไม่มีช่วงก่อนหน้าให้เทียบ
        assert not at.metric[0].delta


def test_overview_all_period_revenue_matches_database(data):
    at = page("overview").run()
    at.selectbox(key="ov_period").set_value("ทั้งหมด").run()
    total = data.sql("SELECT SUM(line_total) AS t FROM fact_sales")["t"].iloc[0]
    assert at.metric[0].value == f"฿{total:,.0f}"


@pytest.mark.parametrize("branch", ["nimman", "cmu", "thaphae"])
def test_dashboard_each_branch(data, branch):
    from manager.theme import BRANCH_TH
    at = page("dashboard").run()
    at.session_state["db_branch"] = branch
    at.run()
    assert not at.exception
    assert any(md.value == f"### สาขา{BRANCH_TH[branch]}" for md in at.markdown)


def test_dashboard_menu_filter_and_show_all(data):
    at = page("dashboard").run()
    assert len(at.dataframe[0].value) == 6                     # ปิดรายละเอียด: แสดง 6 อันดับแรก
    at.toggle(key="db_menu_all").set_value(True).run()
    n_products = data.sql("SELECT COUNT(DISTINCT product_id) n FROM fact_sales")["n"].iloc[0]
    assert len(at.dataframe[0].value) == n_products
    at.session_state["db_cat"] = "เบเกอรี่"
    at.run()
    assert set(at.dataframe[0].value["category"]) == {"เบเกอรี่"}


def test_upload_page_steps_and_show_all(data):
    at = page("upload").run()
    titles = [e.label for e in at.expander]
    assert [t.split()[0] for t in titles] == ["①", "②", "③", "④", "⑤"]
    assert "ตรวจยอดผ่าน" in titles[1]                           # สรุปผลอยู่บนหัวกล่องแม้ยังไม่เปิด
    at.toggle(key="up_show_all").set_value(True).run()
    assert not at.exception
    assert all(e.proto.expanded for e in at.expander)


def test_overview_daily_table_toggle(data):
    at = page("overview").run()
    n = len(at.dataframe)
    at.toggle(key="ov_table").set_value(True).run()
    assert not at.exception and len(at.dataframe) == n + 1


def test_forecast_method_toggle(data):
    at = page("forecast").run()
    at.toggle(key="fc_method").set_value(True).run()
    assert not at.exception
    assert any("Nested walk-forward" in m.value for m in at.markdown)


# ---- นำเข้าไฟล์ ----
def test_import_loads_upload_without_touching_sample(data, tmp_path):
    from manager import importer
    files = raw_files(data)
    with closing(sqlite3.connect(data.DEFAULT_DB)) as con:
        sample_rows = con.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0]
    folder = importer.stage(files, tmp_path / "staged")
    checks = importer.run_checks(folder)
    assert checks["quality_after"].overall["score_avg"] > checks["quality_before"].overall["score_avg"]
    result = importer.load_upload(files, checks)
    assert result["verification"]["passed"].all()
    assert data.UPLOAD_DB.exists()
    with closing(sqlite3.connect(data.DEFAULT_DB)) as con:
        assert con.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0] == sample_rows

    at = page("overview")                                       # หน้าเว็บอ่านชุดข้อมูลที่อัปโหลดได้
    at.session_state["dataset"] = "upload"
    at.run()
    assert not at.exception


# ---- ฟังก์ชันคำนวณ (ไม่ต้องใช้ฐานข้อมูล) ----
def test_detect_kind_recognises_every_raw_file(data):
    for kind, (name, _) in data.UPLOAD_SPEC.items():
        r = insights.detect_kind(name, (data.DEFAULT_RAW / name).read_bytes(), data.UPLOAD_SPEC)
        assert r["kind"] == kind and r["problem"] is None and r["rows"] > 0


def test_detect_kind_reports_missing_columns_and_bad_files(data):
    broken = pd.DataFrame({"receipt_no": ["1"], "item_code": ["C01"]}).to_csv(index=False).encode()
    r = insights.detect_kind("old_export.csv", broken, data.UPLOAD_SPEC)
    assert r["kind"] is None and "qty" in r["problem"]
    assert insights.detect_kind("report.pdf", b"%PDF", data.UPLOAD_SPEC)["problem"].startswith("รองรับเฉพาะ")
    assert "อ่านไฟล์ไม่ได้" in insights.detect_kind("x.json", b"{not json", data.UPLOAD_SPEC)["problem"]


def test_period_bounds_and_change():
    last, first = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-01-01")
    b = insights.period_bounds(last, 7, first)
    assert b["start"] == pd.Timestamp("2026-06-24") and b["prev_end"] == pd.Timestamp("2026-06-23")
    assert (b["prev_end"] - b["prev_start"]).days == 6
    assert insights.period_bounds(last, None, first)["prev_start"] is None
    assert insights.change(110, 100) == pytest.approx(0.1)
    assert insights.change(5, 0) is None and insights.change(5, None) is None


def test_kpi_changes_member_share_in_points():
    cur = pd.DataFrame({"receipt_id": ["a", "b"], "line_total": [100, 100], "qty": [1, 1], "member_id": ["M1", None]})
    prev = pd.DataFrame({"receipt_id": ["c", "d"], "line_total": [50, 50], "qty": [1, 1], "member_id": [None, None]})
    k = insights.kpi_changes(cur, prev)
    assert k["revenue"] == (200, pytest.approx(1.0))
    assert k["member_share"] == (0.5, pytest.approx(50.0))     # 0% -> 50% = +50 จุด
    assert cur["member_id"].tolist() == ["M1", None]           # ไม่แก้ข้อมูลที่รับเข้ามา


def test_day_type_avg_ignores_missing_days():
    d = pd.DataFrame({"revenue": [100, 200, 300, None], "data_missing": [0, 0, 0, 1],
                      "is_weekend": [0, 1, 0, 0], "is_holiday": [0, 0, 1, 0]})
    out = insights.day_type_avg(d)
    assert out["day_type"].tolist() == ["วันธรรมดา", "เสาร์–อาทิตย์", "วันหยุดนักขัตฤกษ์"]
    assert out["days"].sum() == 3
