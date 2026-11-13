"""توليد التقارير القابلة للتنزيل (HTML قابل للطباعة كـ PDF + Excel)."""
from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd

CSS = """
body{font-family:Tajawal,Arial,sans-serif;direction:rtl;color:#1E293B;max-width:1000px;margin:auto;padding:24px;line-height:1.7}
h1{color:#0B1F3A;border-bottom:3px solid #1F5EFF;padding-bottom:8px} h2{color:#0B1F3A;margin-top:32px;border-right:5px solid #1F5EFF;padding-right:10px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px} th{background:#0B1F3A;color:#fff;padding:7px}
td{border-bottom:1px solid #E2E8F0;padding:6px;text-align:center} tr:nth-child(even) td{background:#F8FAFC}
.kpis{display:flex;gap:10px;flex-wrap:wrap} .k{flex:1;min-width:150px;border:1px solid #E2E8F0;border-top:3px solid #1F5EFF;border-radius:10px;padding:10px}
.k b{display:block;font-size:22px;color:#0B1F3A} .warn{background:#FFFBEB;border-right:4px solid #F59E0B;padding:10px;border-radius:6px}
.meta{color:#64748B;font-size:12px} @media print{.pagebreak{page-break-before:always}}
"""


def _table(df: pd.DataFrame, n: int = 25) -> str:
    return df.head(n).to_html(index=False, border=0, float_format=lambda x: f"{x:,.2f}")


def build_html(title: str, sections: list[tuple[str, str]], demo: bool) -> str:
    body = [f"<h1>{html.escape(title)}</h1>",
            f"<p class='meta'>AI Labor Market Intelligence • تاريخ التوليد: {datetime.now():%Y-%m-%d %H:%M}</p>"]
    if demo:
        body.append("<p class='warn'>⚠️ هذا التقرير مبني على بيانات تجريبية (DEMO) لاختبار النظام، وأرقامه ليست إحصائيات حقيقية لسوق العمل.</p>")
    for h, content in sections:
        body.append(f"<h2>{html.escape(h)}</h2>{content}")
    return (f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<link href='https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap' rel='stylesheet'>"
            f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")


def kpi_html(items: list[tuple[str, str]]) -> str:
    return "<div class='kpis'>" + "".join(f"<div class='k'>{html.escape(a)}<b>{html.escape(str(b))}</b></div>" for a, b in items) + "</div>"


def build_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            d = df.copy()
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    d[c] = d[c].dt.strftime("%Y-%m-%d")
            d.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()
