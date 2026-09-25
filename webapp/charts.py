"""
ฟังก์ชันวาดกราฟ (plotly) ใช้เทคนิคการแสดงผลจาก Ch6 และ Lab 8-9
    Line chart + Moving average, Bar / Stacked bar, Heatmap, Boxplot, Histogram, Scatter
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BRANCH_COLORS = {"นิมมาน": "#2a9d8f", "มช.": "#e76f51", "ท่าแพ": "#264653"}
FONT = dict(family="Sarabun, Tahoma, sans-serif")


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), font=FONT, legend_title_text="")
    return fig


def line_with_moving_average(df: pd.DataFrame, x: str, y: str, color: str, window: int, title: str) -> go.Figure:
    """เส้นจาง = ค่ารายวัน / เส้นเข้ม = Moving average (ลดความผันผวนระยะสั้น เห็นแนวโน้ม Ch6)"""
    fig = go.Figure()
    for name, g in df.groupby(color):
        g = g.sort_values(x)
        c = BRANCH_COLORS.get(name)
        fig.add_scatter(x=g[x], y=g[y], mode="lines", name=f"{name} (รายวัน)", line=dict(color=c, width=1),
                        opacity=0.35)
        fig.add_scatter(x=g[x], y=g[y].rolling(window, min_periods=1).mean(), mode="lines",
                        name=f"{name} (เฉลี่ย {window} วัน)", line=dict(color=c, width=3))
    fig.update_layout(title=title, yaxis_title="บาท")
    return _style(fig)


def lines(df: pd.DataFrame, x: str, y: str, color: str, title: str, y_title: str = "") -> go.Figure:
    fig = px.line(df.sort_values(x), x=x, y=y, color=color, title=title, color_discrete_map=BRANCH_COLORS)
    fig.update_layout(yaxis_title=y_title)
    return _style(fig)


def bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, barmode: str = "group",
        orientation: str = "v", text_auto: bool | str = False, height: int = 380) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title, barmode=barmode, orientation=orientation,
                 text_auto=text_auto, color_discrete_map=BRANCH_COLORS)
    return _style(fig, height)


def heatmap(pivot: pd.DataFrame, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrBr", title=title, text_auto=True)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return _style(fig, 420)


def box(df: pd.DataFrame, x: str, y: str, title: str, log_y: bool = False) -> go.Figure:
    fig = px.box(df, x=x, y=y, title=title, log_y=log_y, points="outliers")
    return _style(fig)


def strip(df: pd.DataFrame, x: str, y: str, color: str, title: str, log_y: bool = False) -> go.Figure:
    fig = px.strip(df, x=x, y=y, color=color, title=title, log_y=log_y)
    return _style(fig)


def histogram(df: pd.DataFrame, x: str, title: str, color: str | None = None, nbins: int = 30) -> go.Figure:
    fig = px.histogram(df, x=x, color=color, nbins=nbins, title=title, color_discrete_map=BRANCH_COLORS,
                       barmode="overlay", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash")
    return _style(fig)


def actual_vs_predicted(df: pd.DataFrame, methods: dict[str, str], title: str) -> go.Figure:
    """ยอดจริง (เส้นดำ) เทียบค่าที่ทายของแต่ละวิธี"""
    fig = go.Figure()
    fig.add_scatter(x=df["date"], y=df["revenue"], mode="lines+markers", name="ยอดจริง",
                    line=dict(color="black", width=3))
    for col, label in methods.items():
        fig.add_scatter(x=df["date"], y=df[col], mode="lines", name=label, line=dict(dash="dot"))
    fig.update_layout(title=title, yaxis_title="บาท")
    return _style(fig)
