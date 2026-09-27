# ☕ Doi Brew ETL: รวมข้อมูลยอดขายร้านกาแฟ 3 สาขา

โครงงานวิชา **204426 Data Engineering** ภาคเรียนที่ 1/2569

| | |
|---|---|
| **ชื่อกลุ่ม** | _(เติมชื่อกลุ่มภาษาอังกฤษ)_ |
| **สมาชิก** | 1. _(รหัส)_ _(ชื่อ-นามสกุล)_ <br> 2. _(รหัส)_ _(ชื่อ-นามสกุล)_ <br> 3. _(รหัส)_ _(ชื่อ-นามสกุล)_ |

## โปรเจกต์นี้คืออะไร

ร้านกาแฟ **Doi Brew** (ข้อมูลสมมุติ) มี 3 สาขาในเชียงใหม่ ได้แก่ **นิมมาน, มช., ท่าแพ** แต่ละสาขาใช้ระบบ POS คนละยี่ห้อ ข้อมูลจึงออกมาคนละรูปแบบ (CSV / Excel / JSON) และมีปัญหาคุณภาพข้อมูลหลายแบบ เช่น ข้อมูลซ้ำ ค่าว่าง ชื่อเมนูสะกดผิด จำนวนพิมพ์ผิด

โปรแกรมนี้เป็น **ETL Pipeline** ที่รวม ตรวจคุณภาพ ทำความสะอาด แปลง และโหลดข้อมูลเข้าฐานข้อมูล SQLite แล้วนำไปแสดงผลบนหน้าเว็บ พร้อมโมเดลแนะนำเมนูและทำนายยอดขาย

```
① ข้อมูลดิบ 3 รูปแบบ → ② Extract + Integration → ③ Data Quality (6 มิติ) → ④ Cleaning
   → ⑤ Transform → ⑤ Load (SQLite) → Dashboard / Recommendation / Forecast
```

| ขั้น | เทคนิคหลัก | บท |
|---|---|---|
| Extract + Integration | Schema mapping, JSON → ตาราง, แปลงวันเวลา, Translation mapping, ตรวจยอด | Ch1, Ch2, Ch5 |
| Data Quality | กฎคุณภาพ 21 ข้อ 6 มิติ, Simple ratio, Weighted average, Min operation | Ch3 |
| Cleaning | Deduplication, Validity rules, **เทียบ Outlier 4 วิธี**, **เทียบวิธีเติมค่าว่าง 5 วิธี** | Ch3, Ch4 |
| Transform | Enrichment, Discretization, Anonymization, Encoding, Normalization, Star schema | Ch2, Ch5 |
| Load | Primary/Foreign key, CHECK, Initial / Full refresh / Incremental loading, Load verification | Ch1 |
| แสดงผล | Line + Moving average, Heatmap, Bar, Boxplot, Histogram | Ch6 |
| โมเดล (โบนัส) | Association rules (Recommendation), Linear Regression (Forecast) | – |

## การติดตั้ง

ต้องมี **Python 3.11 ขึ้นไป** (ทดสอบแล้วกับ Python 3.13 และ 3.11 บน Windows)

```bash
# 1) เข้าโฟลเดอร์โปรเจกต์ (โฟลเดอร์ที่แตกจาก zip ซึ่งมีไฟล์ README.md นี้)
cd "<โฟลเดอร์ที่แตกจาก zip>"

# 2) (แนะนำ) สร้างสภาพแวดล้อมแยก เพื่อไม่ให้กระทบไลบรารีอื่นในเครื่อง
py -3.13 -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate

# 3) ติดตั้งไลบรารี
python -m pip install -r requirements.txt
```

> ในเครื่องที่ติดตั้ง Python ไว้หลายเวอร์ชันบน Windows ใช้ `py -3.13` แทน `python` ได้ (ถ้าไม่ได้สร้าง `.venv`)

## การรัน

### 1) หน้าเว็บ (แนะนำ)
```bash
python -m streamlit run app.py
```
แล้วเปิด **http://localhost:8501**
- ถ้ายังไม่มีฐานข้อมูล (เช่น เพิ่งแตกไฟล์ zip ที่ไม่มีฐานข้อมูล) หน้าเว็บจะ**รัน pipeline ให้อัตโนมัติ**ครั้งแรก รอประมาณ 20–60 วินาที
- แถบด้านซ้าย: **🔄 รัน pipeline ใหม่** (รันทุกขั้นพร้อมแถบความคืบหน้า) / **📤 อัปโหลดไฟล์ใหม่** (ต้องเป็นข้อมูลช่วง ม.ค.–มิ.ย. 2569)
- หน้า ⑤ มี**ช่องพิมพ์ SQL** (อ่านอย่างเดียว) ลองถามข้อมูลได้เอง

### 1.1) หน้าเว็บ Doi Brew Manager (สำหรับใช้งานจริงของร้าน)
```bash
python -m streamlit run manager_app.py --server.port 8502 --theme.primaryColor "#1B1916"
```
แล้วเปิด **http://localhost:8502** (รันพร้อมหน้าเว็บเดิมที่ 8501 ได้ ใช้ฐานข้อมูลเดียวกัน)
- 4 หน้า (เมนูด้านบน): **อัปโหลดไฟล์** / **ภาพรวม** / **แดชบอร์ด** / **คาดการณ์**
- หน้าอัปโหลดไฟล์มีรายละเอียดขั้นตอน ①–⑤ ของหน้าเว็บเดิม แบบกดเปิด-ปิดทีละขั้น หรือเปิดทั้งหมดด้วยสวิตช์
- ทุกการ์ดมี**ป้ายเทคนิค** (เช่น `Ch6 · Moving average`) ปิดได้ที่แถบบนสุดเมื่อต้องการหน้าจอแบบใช้งานจริง

### 2) รัน pipeline ทั้งสายผ่านหน้าจอคำสั่ง
```bash
python -m pipeline.run_all                # รันโมดูล 2 → 8 แล้วพิมพ์สรุปผล
python -m pipeline.run_all --regenerate   # สร้างข้อมูลดิบใหม่ (โมดูล 1) ก่อน แล้วรันทุกขั้น
```
รันทีละขั้นได้ เช่น `python -m pipeline.extract`, `python -m pipeline.quality`, `python -m pipeline.clean`, `python -m pipeline.transform`, `python -m pipeline.load`, `python -m pipeline.recommend`, `python -m pipeline.forecast`

### 3) ทดสอบ
```bash
python -m pytest tests            # tests อัตโนมัติทุกโมดูล (ประมาณ 3–10 นาที ขึ้นกับ RAM ที่ว่าง)
```

### 4) วัดความแม่นเทียบกับเฉลย
ข้อมูลเป็นข้อมูลจำลอง จึงมีเฉลย (`data/generator_log/`) ใช้วัดว่า pipeline แก้ข้อมูลถูกต้องแค่ไหน
```bash
python -m evaluation.evaluate_cleaning     # ตัดแถวถูกไหม / เติมค่าถูกกี่ %
python -m evaluation.evaluate_recommend    # หาคู่เมนูที่ใส่ไว้เจอครบไหม
```
> pipeline และหน้าเว็บ**ไม่อ่านไฟล์เฉลย** (มี test ตรวจ) ใช้เฉพาะใน `tests/` และ `evaluation/`

## โครงสร้างโฟลเดอร์

```
├── README.md, requirements.txt, CLAUDE.md (ข้อตกลงการพัฒนา)
├── data_generator/generate.py   [โมดูล 1] สร้างข้อมูลจำลอง + ใส่ปัญหาที่เจอได้จริง (seed คงที่)
├── data/
│   ├── raw/                     ข้อมูลดิบ: nimman_sales.csv, cmu_sales.xlsx, thaphae_sales.json,
│   │                            members.csv, products.csv, holidays.csv
│   ├── generator_log/           เฉลย (รายการปัญหาที่ใส่ + ข้อมูลสะอาด) ใช้วัดผลเท่านั้น
│   └── processed/doibrew.db     ฐานข้อมูล SQLite (สร้างจาก pipeline)
├── pipeline/                    ส่วน Data Engineering ทั้งหมด (Python + pandas)
│   ├── extract.py               [โมดูล 2] อ่านไฟล์ + Integration → ตาราง staging
│   ├── quality.py               [โมดูล 3] วัดคุณภาพ 6 มิติ
│   ├── clean.py                 [โมดูล 4] ทำความสะอาด
│   ├── transform.py             [โมดูล 5] แปลงข้อมูล + Star schema
│   ├── load.py                  [โมดูล 6] โหลดเข้า SQLite + ตรวจหลังโหลด
│   ├── recommend.py             [โมดูล 7] แนะนำเมนู
│   ├── forecast.py              [โมดูล 8] ทำนายยอดขาย
│   └── run_all.py               [โมดูล 10] รันทุกขั้นด้วยคำสั่งเดียว
├── app.py + webapp/             [โมดูล 9] หน้าเว็บ Streamlit 9 หน้า
├── manager_app.py + manager/    หน้าเว็บ Doi Brew Manager 4 หน้า (ใช้งานจริง เรียกฟังก์ชันเดียวกับ pipeline/)
├── evaluation/                  วัดความแม่นเทียบกับเฉลย
├── tests/                       tests อัตโนมัติ (pytest)
├── scripts/make_submission.py   สร้างไฟล์ส่งงาน ชื่อกลุ่ม.zip
└── docs/
    ├── data_issues.md           ปัญหาที่ใส่ในข้อมูล 25 ประเภท
    └── report_notes/            บันทึกสำหรับเล่มรายงาน (เริ่มอ่านที่ 00_overview.md)
```

## ข้อมูล

ข้อมูลจำลองทั้งหมด (ไม่มีข้อมูลส่วนบุคคลจริง) ช่วง 1 ม.ค.–30 มิ.ย. 2569 หลังทำความสะอาดเหลือประมาณ 11,600 บิล (16,000 รายการสินค้า) สมาชิก 800 คน มีปัญหาที่ใส่ไว้ 25 ประเภท แบ่งเป็น
- **กลุ่ม A** รูปแบบต่างกันทั้งไฟล์ (ชื่อคอลัมน์, วันที่, รหัสสินค้า) → แก้ในขั้น Extract
- **กลุ่ม B** คุณภาพข้อมูล (ซ้ำ, ว่าง, ผิดกฎ) → วัดในขั้น Quality แล้วแก้ในขั้น Cleaning
- **กลุ่ม C** Outlier (พิมพ์ผิด 2 แบบ + ออเดอร์ใหญ่จริงที่ห้ามลบ)

รายละเอียดใน `docs/data_issues.md`

## แก้ปัญหาที่พบบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `database is locked` | มีโปรแกรมอื่นเปิดไฟล์ `doibrew.db` อยู่ (เช่น DB Browser for SQLite) ปิดโปรแกรมนั้นก่อนรัน |
| หน้าเว็บหรือ tests ช้ามาก / ถูกปิดกลางทาง | หน่วยความจำ (RAM) ไม่พอ ปิดโปรแกรมอื่นก่อน (เคยเกิดในเครื่องที่ RAM ว่างน้อยกว่า 2 GB) |
| `Port 8501 is already in use` | มีหน้าเว็บเปิดอยู่แล้ว เปิด http://localhost:8501 ได้เลย หรือรันด้วย `--server.port 8502` |
| เปิดไฟล์ CSV ใน Excel แล้วภาษาไทยเพี้ยน | ไฟล์ของโปรเจกต์บันทึกแบบ UTF-8 พร้อม BOM แล้ว ถ้ายังเพี้ยนให้เปิดผ่าน Data → From Text/CSV แล้วเลือก UTF-8 |
| `ModuleNotFoundError` | ยังไม่ได้ติดตั้งไลบรารี หรือไม่ได้รันคำสั่งจากโฟลเดอร์โปรเจกต์ (ต้องอยู่โฟลเดอร์เดียวกับ README.md) |
