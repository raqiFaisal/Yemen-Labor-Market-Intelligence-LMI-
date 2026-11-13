"""اختبار شامل لجامع YemenHR على خادم محلي يحاكي بنية الموقع المرصودة.

لا يُستخدم أي ناتج من هذا الاختبار كبيانات: يعمل في مجلد مؤقت ويُحذف.
    python -m tests.test_scraper_mock
يتحقق من: الترقيم عبر Next، ربط التصنيفات، تسجيل الفشل (404 + فشل التحليل)، عدم احتساب الروابط المكتشفة
كإعلانات، إزالة التكرار، والاستكمال دون إعادة التنزيل.
"""
import http.server
import shutil
import tempfile
import threading
from pathlib import Path

import pandas as pd

import scripts.scrapers.base as base
import scripts.scrapers.yemenhr as yh

ROW = ('<tr><td>{p}</td><td><a href="#">{org}</a></td><td><a href="/jobs/{slug}-{id}">{t}</a></td>'
       '<td><a href="#">{loc}</a></td><td>{d}</td><td>Login to mark</td></tr>')
JOBS = [("acc", "aaaaaaa1", "Accountant", "Org A", "Aden", "01 Nov, 26", "15 Nov, 26"),
        ("dev", "aaaaaaa2", "Web Developer", "Org B", "Sana'a", "02 Nov, 26", "16 Nov, 26"),
        ("nurse", "aaaaaaa3", "Nurse", "Org C", "Ma'rib", "03 Nov, 26", "17 Nov, 26"),
        ("gone", "aaaaaaa4", "Removed Job", "Org D", "Taiz", "03 Nov, 26", "17 Nov, 26"),   # 404
        ("broken", "aaaaaaa5", "Broken Page", "Org E", "Ibb", "03 Nov, 26", "17 Nov, 26")]  # بلا وصف


UUID_TAIL = "-ed38-4e8c-bc95-617a6c6ef025"
# صفحات التصنيفات في الموقع الحقيقي تربط الإعلانات بصيغة UUID لا بالرابط المختصر (تم التحقق من yemenhr.com)
ROW_UUID = ('<tr><td>{p}</td><td><a href="#">{org}</a></td><td><a href="/jobs/{id}' + UUID_TAIL + '">{t}</a></td>'
            '<td><a href="#">{loc}</a></td><td>{d}</td></tr>')


def page(rows, nxt=None, uuid=False):
    n = f'<a href="{nxt}">Next</a>' if nxt else '<a href="#">Next</a>'
    tpl = ROW_UUID if uuid else ROW
    body = "".join(tpl.format(slug=s, id=i, t=t, org=o, loc=l, p=p, d=d) for s, i, t, o, l, p, d in rows)
    cats = '<a href="/categories/finance-accounting">Finance/Accounting 1</a><a href="/categories/healthcare">Healthcare 1</a>'
    return f"<html><body><div>Total: 5</div>First Prev {n}<table>{body}</table>{cats}</body></html>"


def detail(t, org, loc, broken=False):
    desc = "" if broken else ("<p>MAIN DESCRIPTION</p><p>Prepare monthly financial reports and reconcile bank accounts "
                              "using Excel and ERP.</p><p>REQUIREMENTS</p><ul><li>Bachelor degree</li>"
                              "<li>3 years of experience</li></ul>")
    return (f"<html><body><main><h1>{t}</h1><div>{org}</div><div>{loc}</div><div>Posted: 01 Nov, 26</div>"
            f"<div>Deadline: 15 Nov, 26</div><h3>Job Description /الوصف الوظيفي</h3>{desc}"
            f"<h3>How to Apply /كيفية التقديم</h3><p>email</p></main></body></html>")


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path
        routes = {"/robots.txt": "User-agent: *\nAllow: /\n", "/jobs": page(JOBS[:3], "/jobs?pg=2"),
                  "/jobs?pg=2": page(JOBS[3:]), "/categories/finance-accounting": page(JOBS[:1], uuid=True),
                  "/categories/healthcare": page(JOBS[2:3], uuid=True)}
        for s, i, t, o, l, *_ in JOBS:
            if s != "gone":
                routes[f"/jobs/{s}-{i}"] = detail(t, o, l, broken=(s == "broken"))
        if p in routes:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(routes[p].encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    tmp = Path(tempfile.mkdtemp())
    for d in ("logs", "state"):
        (tmp / d).mkdir()
    base.RAW_DIR, base.LOGS_DIR, base.STATE_DIR = tmp, tmp / "logs", tmp / "state"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yh.BASE = f"http://127.0.0.1:{srv.server_port}"
    yh.LIST_URL, yh.PAGE_TEMPLATE = yh.BASE + "/jobs", yh.BASE + "/jobs?page={p}"
    try:
        st0 = yh.run(max_pages=5, delay=0.01, categories=False)   # محاكاة تشغيل سابق بلا تصنيفات
        d0 = pd.read_csv(tmp / "yemen_yemenhr.csv", dtype=str)
        assert d0["category"].fillna("").eq("").all() and st0.downloaded_ok == 3
        (tmp / "yemen_yemenhr.csv").unlink()
        st1 = yh.run(max_pages=5, delay=0.01, debug=False)
        data = pd.read_csv(tmp / "yemen_yemenhr.csv", dtype=str)
        fails = pd.read_csv(tmp / "logs" / "yemenhr_failures.csv", dtype=str)
        assert st1.links_discovered == 5, st1        # روابط UUID في التصنيفات لا تُعد إعلانات جديدة
        assert st1.jobs_with_category == 2, st1
        assert st0.failed == 2, st0                    # 404 + فشل تحليل
        assert st1.downloaded_ok == 3 and st1.failed == 1, st1  # 404 دائم يُتخطى، فشل التحليل يُعاد
        assert len(data) == 3 and data.source_job_id.is_unique
        assert set(fails.stage) == {"detail", "parse"} and fails.reason.str.contains("404").any()
        acc = data[data.title == "Accountant"].iloc[0]
        assert acc.category == "Finance/Accounting" and acc.posted_date == "2026-11-01" and acc.governorate == "عدن"
        st2 = yh.run(max_pages=5, delay=0.01)          # استكمال
        assert st2.downloaded_ok == 0 and st2.already_downloaded == 3, st2
        assert st2.details_attempted == 1, "404 يُتخطى، فشل التحليل يُعاد"  # broken يُعاد، gone لا
        assert len(pd.read_csv(tmp / "yemen_yemenhr.csv")) == 3
        # إعادة ربط التصنيف لإعلانات محفوظة سابقاً دون إعادة تنزيلها
        dd = pd.read_csv(tmp / "yemen_yemenhr.csv", dtype=str)
        dd["category"] = ""
        dd.to_csv(tmp / "yemen_yemenhr.csv", index=False)
        st3 = yh.run(max_pages=5, delay=0.01)
        d3 = pd.read_csv(tmp / "yemen_yemenhr.csv", dtype=str).fillna("")
        assert st3.downloaded_ok == 0 and set(d3.category) == {"", "Finance/Accounting", "Healthcare"}, d3.category.tolist()
        print("✔ جميع اختبارات الجامع نجحت:", {k: v for k, v in vars(st1).items() if isinstance(v, int)})
    finally:
        srv.shutdown()
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
