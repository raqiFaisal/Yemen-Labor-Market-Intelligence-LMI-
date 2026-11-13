"""اختبار جامع telegra.ph على استجابات مصطنعة (بنية واجهة Telegraph الرسمية) — بلا إنترنت، في مجلد مؤقت.
    python -m tests.test_telegraph_collector
نتائجه ليست نتائج جمع حقيقية.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import scripts.collectors.telegraph as tp
import scripts.collectors.telegram_web as tw

PAGES = {
    "Accountant-11-05": {"ok": True, "result": {"title": "Accountant", "author_name": "HR Yemen", "views": 12, "content": [
        {"tag": "h4", "children": ["المسؤوليات"]},
        {"tag": "ul", "children": [{"tag": "li", "children": ["إعداد القوائم المالية"]},
                                   {"tag": "li", "children": ["Excel ", {"tag": "strong", "children": ["متقدم"]}]}]},
        {"tag": "p", "children": ["للتقديم: ", {"tag": "a", "attrs": {"href": "https://forms.gle/x"}, "children": ["هنا"]}]},
        {"tag": "p", "children": ["سطر", {"tag": "br"}, "ثانٍ"]}]}},
    "Gone-01-01": {"ok": False, "error": "PAGE_NOT_FOUND"},
    "Nurse-11-04": {"ok": True, "result": {"title": "Nurse", "content": [{"tag": "p", "children": ["تمريض — عدن"]}]}},
}
HTML_ONLY = {"Html-Only-11-03": '<article id="_tl_editor" class="tl_article_content"><h1>Engineer</h1><address>Org</address>'
                                '<p>خبرة 3 سنوات</p><p>AutoCAD<br>Revit</p></article>'}


class Fake:
    calls, flaky = [], set()

    def __init__(self, *a, **k):
        pass

    def get(self, url):
        Fake.calls.append(url)
        path = url.split("/getPage/")[-1].split("?")[0] if "getPage" in url else url.rsplit("/", 1)[-1]
        if path in Fake.flaky:
            return None, "HTTP 503", 503
        if "getPage" in url:
            if path in PAGES:
                return json.dumps(PAGES[path]), None, 200
            return None, "HTTP 500", 500
        if path in HTML_ONLY:
            return HTML_ONLY[path], None, 200
        return None, "HTTP 404", 404


def check(c, label, R):
    R.append(bool(c))
    print(("✔ " if c else "✖ ") + label)


def main():
    R, tmp = [], Path(tempfile.mkdtemp(prefix="lmi_tp_test_"))
    orig = (tw.TG_DIR, tp.TG_DIR, tp.OUT_DIR, tp.LOGS_DIR, tp.Fetcher)
    tp.TG_DIR = tmp / "telegram"
    tp.OUT_DIR, tp.LOGS_DIR, tp.Fetcher = tp.TG_DIR / "telegraph", tmp / "logs", Fake
    tp.TG_DIR.mkdir(parents=True)
    try:
        check(tp.path_of("https://telegra.ph/Accountant-11-05?x=1") == "Accountant-11-05", "استخراج المسار", R)
        check(tp.path_of("https://telegra.ph/file/abc.jpg") is None and tp.path_of("https://linkedin.com/x") is None,
              "يتجاهل الصور والمواقع الأخرى", R)
        rows = [  # منشوران يشيران لنفس الصفحة + صفحة محذوفة + HTML فقط + فشل مؤقت + منشور بلا رابط
            (10, "2026-11-05T00:00:00+00:00", "https://telegra.ph/Accountant-11-05 https://www.linkedin.com/jobs/1"),
            (9, "2026-11-04T00:00:00+00:00", "https://telegra.ph/Accountant-11-05"),
            (8, "2026-11-04T00:00:00+00:00", "https://telegra.ph/Nurse-11-04"),
            (7, "2026-11-03T00:00:00+00:00", "https://telegra.ph/Gone-01-01"),
            (6, "2026-11-03T00:00:00+00:00", "https://telegra.ph/Html-Only-11-03"),
            (5, "2026-11-02T00:00:00+00:00", "https://telegra.ph/Flaky-11-02"),
            (4, "2026-11-01T00:00:00+00:00", "")]
        pd.DataFrame([{c: "" for c in tw.COLUMNS} | {"source_message_id": m, "message_date": d, "link_urls": l}
                      for m, d, l in rows]).to_csv(tp.TG_DIR / "chan.csv", index=False)
        PAGES["Flaky-11-02"] = {"ok": True, "result": {"title": "Flaky", "content": ["نص"]}}
        Fake.flaky = {"Flaky-11-02"}

        st = tp.run(["chan"], delay=0)
        p = pd.read_csv(tp.OUT_DIR / "pages.csv", dtype=str, keep_default_na=False).set_index("telegraph_path")
        l = pd.read_csv(tp.OUT_DIR / "links.csv", dtype=str)
        check(st.unique_pages_linked == 5 and st.posts_with_telegraph == 6, "5 صفحات فريدة من 6 منشورات", R)
        check(st.saved_ok == 3 and st.via_api == 2 and st.via_html == 1, "حُفظت 3 (API 2 + HTML احتياطي 1)", R)
        t = p.loc["Accountant-11-05", "content_text"]
        check("المسؤوليات" in t and "- إعداد القوائم المالية" in t and "Excel متقدم" in t and "سطر\nثانٍ" in t,
              "النص الكامل مع البنية (عناوين/قوائم/أسطر)", R)
        check(p.loc["Accountant-11-05", "content_links"] == "https://forms.gle/x", "روابط الصفحة", R)
        check(p.loc["Accountant-11-05", "first_message_id"] == "10", "الربط بأحدث منشور أصلي", R)
        check(set(l[l.telegraph_path == "Accountant-11-05"].source_message_id) == {"10", "9"},
              "links.csv يحفظ كل المنشورات المشيرة لنفس الصفحة", R)
        check("AutoCAD\nRevit" in p.loc["Html-Only-11-03", "content_text"] and p.loc["Html-Only-11-03", "title"] == "Engineer",
              "الاحتياطي HTML يحفظ النص والعنوان", R)
        fails = pd.read_csv(tp.LOGS_DIR / "telegraph_failures.csv", dtype=str)
        perm = dict(zip(fails.telegraph_path, fails.permanent))
        check(perm.get("Gone-01-01") == "True" and perm.get("Flaky-11-02") == "False", "الفشل الدائم/المؤقت مصنّف", R)

        Fake.calls.clear()
        Fake.flaky = set()
        st = tp.run(["chan"], delay=0)
        urls = " ".join(Fake.calls)
        check(st.attempted == 1 and st.saved_ok == 1 and "Flaky" in urls and "Accountant" not in urls and "Gone" not in urls,
              "الاستكمال: يعيد المؤقت فقط، لا يعيد المحفوظ ولا الدائم", R)
        p2 = pd.read_csv(tp.OUT_DIR / "pages.csv", dtype=str)
        check(len(p2) == 4 and p2.telegraph_path.is_unique and st.remaining == 1, "4 محفوظة بلا تكرار، متبقٍ 1 (دائم)", R)

        Fake.calls.clear()
        st = tp.run(["chan"], delay=0, retry_failed=True)
        check(st.attempted == 1 and "Gone" in " ".join(Fake.calls), "--retry-failed يعيد محاولة الدائم", R)

        st = tp.run(["chan"], limit=0, delay=0)
        check(st.attempted == 0, "--limit 0 لا يجلب شيئاً", R)
    finally:
        tw.TG_DIR, tp.TG_DIR, tp.OUT_DIR, tp.LOGS_DIR, tp.Fetcher = orig
        PAGES.pop("Flaky-11-02", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nالنتيجة: {sum(R)}/{len(R)} (استجابات مصطنعة — ليست نتائج جمع حقيقية)")
    raise SystemExit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
