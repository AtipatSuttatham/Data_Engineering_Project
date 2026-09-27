"""
หน้าตาของ Doi Brew Manager: สี, ตัวอักษร, การ์ด, ป้ายเทคนิค และกราฟ (plotly)

ตั้งค่าผ่าน CSS ใน manager_app.py เท่านั้น ไม่แก้ .streamlit/config.toml เพื่อไม่ให้กระทบหน้าเว็บเดิม (app.py)
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# สีหลัก (ชุดเดียวกับไฟล์ออกแบบ v4: พื้นเทาอุ่น การ์ดขาวเงาบาง สีเน้นเดียว)
BG, SURF, SURF2, LINE = "#F4F3EF", "#FFFFFF", "#F1EFEA", "#E7E4DD"
INK, INK2 = "#1B1916", "#67625A"
PRI, CAR = "#5A3619", "#C8925E"          # กาแฟเข้ม / กาแฟอ่อน (ใช้ในกราฟ)
ACC = "#7A4B24"
GOOD, BAD = "#2F7D4F", "#B3261E"

# สีประจำสาขา: ต่างกันทั้งเฉดสีและความสว่าง ใช้ตรงกันทุกกราฟ
BRANCH_COLOR = {"nimman": "#2F6DB5", "cmu": "#E0912F", "thaphae": "#3FA394"}
BRANCH_TH = {"nimman": "นิมมาน", "cmu": "มช.", "thaphae": "ท่าแพ"}
COLOR_BY_TH = {BRANCH_TH[b]: c for b, c in BRANCH_COLOR.items()}
CATEGORY_TH = {"coffee": "กาแฟ", "non_coffee": "ไม่ใช่กาแฟ", "bakery": "เบเกอรี่"}
CATEGORY_COLOR = {"กาแฟ": PRI, "ไม่ใช่กาแฟ": CAR, "เบเกอรี่": "#E6D3BD"}

FONT = "Anuphan, system-ui, sans-serif"
SHADOW = "0 1px 2px rgba(27,25,22,0.05), 0 0 0 1px #ECE9E3"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, select, textarea {{ font-family: 'Anuphan', system-ui, sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 4rem; }}
[data-testid="stLogo"] {{ height: 2.1rem; }}
[data-testid="stHeader"] {{ background: rgba(255,255,255,0.92); border-bottom: 1px solid {LINE}; }}
h1 {{ font-weight: 700 !important; letter-spacing: -0.02em; color: {INK}; }}
h2, h3, h4 {{ color: {INK}; font-weight: 600 !important; letter-spacing: -0.01em; }}
[class*="st-key-card"] {{ background: {SURF}; border-radius: 20px; padding: 20px 24px 18px; box-shadow: {SHADOW}; }}
[data-testid="stMetric"] {{ background: {SURF}; border-radius: 20px; padding: 16px 20px; box-shadow: {SHADOW};
                           min-height: 134px; box-sizing: border-box; }}
/* การ์ดในแถวเดียวกันสูงเท่ากัน: คอลัมน์ยืดตามแถว และการ์ดใบสุดท้ายของคอลัมน์ยืดเต็มที่ที่เหลือ */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [class*="st-key-card"]) {{ align-items: stretch; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:last-child:has(> [class*="st-key-card"]) {{ flex: 1 1 auto; }}
[data-testid="stLayoutWrapper"] > [class*="st-key-card"] {{ height: 100%; box-sizing: border-box; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; letter-spacing: -0.02em; font-weight: 600; }}
[data-testid="stExpander"] details {{ background: {SURF}; border: none; border-radius: 18px; box-shadow: {SHADOW}; }}
[data-testid="stExpander"] summary {{ padding: 16px 20px; }}
[data-testid="stExpander"] summary p {{ font-size: 17px; font-weight: 600; }}
.tech {{ display: inline-block; padding: 2px 8px; margin: 0 4px 2px 0; border-radius: 6px; background: {SURF2};
         color: {INK2}; font-size: 11.5px; font-weight: 600; white-space: nowrap; }}
.chip {{ display: inline-block; padding: 3px 10px; margin: 0 6px 4px 0; border-radius: 999px; background: {SURF2};
         color: {INK}; font-size: 12.5px; font-variant-numeric: tabular-nums; }}
.sub {{ color: {INK2}; font-size: 13.5px; margin: -6px 0 6px; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ส่วนประกอบ
# ---------------------------------------------------------------------------
def show_tech() -> bool:
    """ป้ายเทคนิค (โหมดนำเสนอ) เปิด/ปิดได้ที่แถบด้านซ้าย"""
    return st.session_state.get("show_tech", True)


def tech(*labels: str) -> None:
    """ป้ายบอกว่าการ์ดนี้ใช้เทคนิคจากบทไหน (ตามขอบเขตใน CLAUDE.md หัวข้อ 3)"""
    if show_tech() and labels:
        chips = "".join(f'<span class="tech">{html.escape(l)}</span>' for l in labels)
        st.markdown(chips, unsafe_allow_html=True)


def chips(*labels: str) -> None:
    """สรุปผลสั้น ๆ ของแต่ละขั้น (แสดงเสมอ แม้ปิดรายละเอียด)"""
    st.markdown("".join(f'<span class="chip">{html.escape(l)}</span>' for l in labels), unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> None:
    """รายการ ชื่อ : ค่า แบบตารางเรียบ (ใช้แทน dataframe ที่ไม่มีหัวคอลัมน์)"""
    body = "".join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;'
                   f'border-top:1px solid {SURF2};font-size:14px"><span style="color:{INK2}">{html.escape(k)}</span>'
                   f'<span style="font-variant-numeric:tabular-nums;text-align:right">{html.escape(v)}</span></div>'
                   for k, v in rows)
    st.markdown(f"<div>{body}</div>", unsafe_allow_html=True)


def card(key: str):
    """การ์ดพื้นขาว (สไตล์มาจาก CSS ที่จับ class st-key-card...)"""
    return st.container(key=f"card_{key}")


def heading(title: str, sub: str | None = None, techs: tuple[str, ...] = ()) -> None:
    st.markdown(f"#### {title}")
    if sub:
        st.markdown(f'<p class="sub">{html.escape(sub)}</p>', unsafe_allow_html=True)
    tech(*techs)


def baht(x: float, digits: int = 0) -> str:
    return f"฿{x:,.{digits}f}"


def delta_text(change: float | None, unit: str = "%") -> str | None:
    """ข้อความเปลี่ยนแปลงสำหรับ st.metric (None = ไม่มีช่วงก่อนหน้าให้เทียบ)"""
    if change is None or pd.isna(change):
        return None
    return f"{change * 100:+.1f}{unit}" if unit == "%" else f"{change:+.1f}{unit}"


# ---------------------------------------------------------------------------
# กราฟ
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def daily_with_ma(daily: pd.DataFrame, window: int) -> go.Figure:
    """เส้นจาง = ยอดจริงรายวัน (วันที่ข้อมูลหายเป็นช่องว่าง ไม่ใช่ 0) / เส้นเข้ม = Moving average (Ch6)"""
    fig = go.Figure()
    for b, g in daily.sort_values("date").groupby("branch", sort=False):
        c, name = BRANCH_COLOR[b], BRANCH_TH[b]
        fig.add_scatter(x=g["date"], y=g["revenue"], mode="lines", line=dict(color=c, width=1), opacity=0.35,
                        name=f"{name} รายวัน", showlegend=False, connectgaps=False)
        fig.add_scatter(x=g["date"], y=g["revenue"].rolling(window, min_periods=1).mean(), mode="lines",
                        line=dict(color=c, width=3), name=name)
    fig.update_yaxes(title_text="บาท", tickformat=",.0f")
    return style(fig)


def hbar(df: pd.DataFrame, x: str, y: str, color: str | list = PRI, text: str | None = None,
         height: int = 260) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x], y=df[y], orientation="h", marker_color=color,
                           text=df[text] if text else None, textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, float(df[x].max()) * 1.45])  # เผื่อที่ให้ตัวเลข
    return style(fig, height, legend=False)


def heatmap(pivot: pd.DataFrame, height: int = 460) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=[f"{h:02d}:00" for h in pivot.index],
                               colorscale=[[0, "#F4EBDD"], [1, PRI]], text=pivot.round(1).values,
                               texttemplate="%{text}", xgap=3, ygap=3, colorbar=dict(title="บิล", thickness=10)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return style(fig, height, legend=False)


def branch_lines(df: pd.DataFrame, x: str, y: str, y_title: str = "", height: int = 340) -> go.Figure:
    fig = go.Figure()
    for b, g in df.sort_values(x).groupby("branch", sort=False):
        fig.add_scatter(x=g[x], y=g[y], mode="lines", line=dict(color=BRANCH_COLOR[b], width=2.5), name=BRANCH_TH[b])
    fig.update_yaxes(title_text=y_title)
    return style(fig, height)
