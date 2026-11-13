"""اختبار تكامل كامل في الوضع الحقيقي على خادم YemenHR وهمي (محلي): جمع → توحيد → مهارات → Embeddings → FNN → LSTM.

⚠️ للاختبار فقط: يكتب إعلانات وهمية في data/raw، لذلك يرفض العمل إلا داخل نسخة مؤقتة من المشروع:
    cp -r lmi /tmp/lmi_e2e && cd /tmp/lmi_e2e && rm -rf data/raw/* && LMI_E2E_ROOT=/tmp/lmi_e2e python -m tests.e2e_mock
"""
import http.server
import random
import threading

from tests.test_scraper_mock import ROW, ROW_UUID

CATS = {"Finance/Accounting": ("Accountant", "Prepare financial reports, IFRS, reconcile bank accounts using Excel and ERP."),
        "Computers/IT": ("Software Developer", "Develop web apps with JavaScript, React, SQL and Git."),
        "Healthcare": ("Nurse", "Patient care, infection control and first aid in the clinic."),
        "Engineering": ("Civil Engineer", "Site supervision, AutoCAD drawings and quantity surveying (BOQ)."),
        "Sales/Marketing": ("Marketing Officer", "Digital marketing, social media management and content writing."),
        "Administration": ("Admin Officer", "Procurement, logistics and report writing with MS Office."),
        "Development": ("Project Officer", "Project management, proposal writing and monitoring and evaluation.")}
CITIES = ["Aden", "Sana'a", "Taiz", "Ma'rib", "Ibb", "Multiple Cities"]


def build(n=84, seed=1):
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        cat = list(CATS)[i % len(CATS)]
        t, d = CATS[cat]
        day = 1 + (i % 28)
        mon = ["Sep", "Oct", "Nov"][i % 3]
        jobs.append({"slug": f"job{i}", "id": f"{i:08x}", "t": f"{t} {i}", "org": f"Org {i % 9}",
                     "loc": rng.choice(CITIES), "p": f"{day:02d} {mon}, 26", "d": f"{min(day + 7, 28):02d} Nov, 26",
                     "cat": cat, "desc": d + (" المهارات: اللغة الإنجليزية والعمل ضمن فريق." if i % 2 else
                                              " Strong communication and English skills required.")})
    return jobs


def server(jobs):
    per = 50

    def page(rows, nxt, uuid=False):
        body = "".join((ROW_UUID if uuid else ROW).format(slug=j["slug"], id=j["id"], t=j["t"], org=j["org"], loc=j["loc"], p=j["p"], d=j["d"]) for j in rows)
        cats = "".join(f'<a href="/categories/{c.lower().replace("/", "-")}">{c} {sum(j["cat"] == c for j in jobs)}</a>' for c in CATS)
        return (f"<html><body>Total: {len(jobs)} <a href=\"{nxt or '#'}\">Next</a><table>{body}</table>{cats}</body></html>")

    routes = {"/robots.txt": "User-agent: *\nAllow: /\n"}
    pages = [jobs[i:i + per] for i in range(0, len(jobs), per)]
    for k, rows in enumerate(pages):
        routes["/jobs" if k == 0 else f"/jobs?pg={k + 1}"] = page(rows, f"/jobs?pg={k + 2}" if k + 1 < len(pages) else None)
    for c in CATS:
        routes[f"/categories/{c.lower().replace('/', '-')}"] = page([j for j in jobs if j["cat"] == c], None, uuid=True)
    for j in jobs:
        routes[f"/jobs/{j['slug']}-{j['id']}"] = (
            f"<html><body><main><h1>{j['t']}</h1><div>{j['org']}</div><div>{j['loc']}</div><div>Posted: {j['p']}</div>"
            f"<div>Deadline: {j['d']}</div><h3>Job Description /الوصف الوظيفي</h3><p>MAIN DESCRIPTION</p><p>{j['desc']}</p>"
            f"<p>REQUIREMENTS</p><ul><li>Bachelor degree</li><li>2 years of experience</li></ul>"
            f"<h3>How to Apply /كيفية التقديم</h3><p>apply</p></main></body></html>")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in routes:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(routes[self.path].encode())
            else:
                self.send_response(404)
                self.end_headers()

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path
    from core.config import ROOT
    if Path(os.environ.get("LMI_E2E_ROOT", "/nonexistent")).resolve() != ROOT.resolve():
        raise SystemExit("مرفوض: هذا الاختبار يكتب بيانات وهمية. شغّله فقط في نسخة مؤقتة مع LMI_E2E_ROOT=<مسار النسخة>.")
    import scripts.scrapers.yemenhr as yh
    srv = server(build())
    yh.BASE = f"http://127.0.0.1:{srv.server_port}"
    yh.LIST_URL, yh.PAGE_TEMPLATE = yh.BASE + "/jobs", yh.BASE + "/jobs?page={p}"
    st = yh.run(max_pages=10, delay=0.01)
    print("SCRAPED_OK", st.downloaded_ok, "FAILED", st.failed, "TOTAL", st.total_in_file)
    srv.shutdown()
    from scripts import train_all
    sys.argv = ["train_all"]
    train_all.main()
