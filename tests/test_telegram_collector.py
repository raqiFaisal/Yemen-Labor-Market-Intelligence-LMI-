"""اختبار جامع Telegram (المرحلة 2) على صفحات HTML مصطنعة بنفس بنية t.me/s/<channel>.

لا يتصل بالإنترنت ولا يُنتج بيانات: يعمل في مجلد مؤقت يُحذف. نتائجه ليست نتائج جمع حقيقية.
    python -m tests.test_telegram_collector
يتحقق من: التحليل (نص/أسطر/روابط/وسائط/معاد توجيه/تاريخ)، الترقيم ?before=، عدم التكرار عند إعادة التشغيل،
التقاط الجديد فقط، رصد تعديل النص مع حفظ النسخة القديمة، اكتمال الأرشيف، وتسجيل الفشل.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import scripts.collectors.telegram_web as tw

CH = "test_jobs"


def msg_html(mid, text, date, edited=False, fwd=None, photo=False):
    t = f'<div class="tgme_widget_message_text js-message_text">{text}</div>' if text else ""
    f = f'<div class="tgme_widget_message_forwarded_from">Forwarded from <a class="tgme_widget_message_forwarded_from_name">{fwd}</a></div>' if fwd else ""
    p = '<a class="tgme_widget_message_photo_wrap" style="background-image:url(x)"></a>' if photo else ""
    e = '<span class="tgme_widget_message_meta">edited <time datetime="x"></time></span>' if edited else \
        '<span class="tgme_widget_message_meta"></span>'
    return (f'<div class="tgme_widget_message_wrap"><div class="tgme_widget_message" data-post="{CH}/{mid}">{f}{p}{t}'
            f'<div class="tgme_widget_message_footer"><span class="tgme_widget_message_views">1.2K</span>{e}'
            f'<a class="tgme_widget_message_date" href="#"><time datetime="{date}" class="time"></time></a></div></div></div>')


class FakeChannel:
    """يحاكي t.me/s: 20 رسالة لكل صفحة، الأحدث أولاً عند before=None، وما قبل N عند ?before=N."""
    def __init__(self, msgs):
        self.msgs, self.calls, self.fail_urls = msgs, [], set()

    def page(self, before):
        ids = sorted(i for i in self.msgs if before is None or i < before)[-20:]
        head = '<div class="tgme_channel_info_header_title">قناة اختبار</div>'
        return "<html><body>" + head + "".join(msg_html(i, *self.msgs[i]) for i in ids) + "</body></html>"


def fake_fetcher_factory(ch: FakeChannel):
    class F:
        def __init__(self, *a, **k):
            pass

        def get(self, url):
            ch.calls.append(url)
            if url in ch.fail_urls:
                return None, "HTTP 503", 503
            before = int(url.split("before=")[1]) if "before=" in url else None
            return ch.page(before), None, 200
    return F


def check(cond, label, results):
    results.append((label, bool(cond)))
    print(("✔ " if cond else "✖ ") + label)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lmi_tg_test_"))
    orig = (tw.TG_DIR, tw.STATE_DIR, tw.LOGS_DIR, tw.Fetcher)
    tw.TG_DIR, tw.STATE_DIR, tw.LOGS_DIR = tmp / "telegram", tmp / "telegram" / "state", tmp / "logs"
    R = []
    try:
        # 1) تحليل صفحة واحدة
        html = "<html>" + msg_html(7, 'مطلوب محاسب<br>عدن<br><a href="https://yemenhr.com/jobs/x">التفاصيل</a>',
                                   "2026-11-01T08:00:00+00:00", edited=True, fwd="YemenHR", photo=True) + \
               msg_html(8, "", "2026-11-02T08:00:00+00:00", photo=True) + \
               '<div class="tgme_widget_message" data-post="other_channel/9"></div></html>'
        posts, oldest, _ = tw.parse_page(html, CH)
        p = posts[0]
        check(len(posts) == 2, "يتجاهل منشورات قنوات أخرى في نفس الصفحة", R)
        check(p["raw_text"] == "مطلوب محاسب\nعدن\nالتفاصيل", "يحفظ النص الخام مع فواصل الأسطر", R)
        check(p["link_urls"] == "https://yemenhr.com/jobs/x", "يستخرج الروابط الخارجية", R)
        check(p["edited"] and p["forwarded_from"] == "YemenHR" and p["has_photo"], "علامات التعديل/التوجيه/الصورة", R)
        check(p["message_date"] == "2026-11-01T08:00:00+00:00" and p["source_url"] == f"https://t.me/{CH}/7",
              "تاريخ الرسالة ورابطها من HTML", R)
        check(posts[1]["media_only"] and oldest == 7, "منشور وسائط بلا نص يُعلَّم media_only", R)

        # 2) أول تشغيل بحد صفحات صغير: لا يكتمل الأرشيف
        msgs = {i: (f"وظيفة رقم {i}", f"2026-{1 + i // 10:02d}-01T00:00:00+00:00") for i in range(1, 56)}
        ch = FakeChannel(msgs)
        tw.Fetcher = fake_fetcher_factory(ch)
        st = tw.run(CH, "all", max_pages=2, delay=0)
        check(st.total_in_file == 40 and not st.archive_complete, "تشغيل أول بحد صفحتين: 40 منشوراً، الأرشيف غير مكتمل", R)

        # 3) استكمال: يكمل من أقدم محفوظ حتى البداية، بلا تكرار
        st = tw.run(CH, "backfill", max_pages=10, delay=0)
        d = pd.read_csv(tw.TG_DIR / f"{CH}.csv", dtype=str)
        check(st.total_in_file == 55 and d.source_message_id.is_unique, "الاستكمال يصل إلى 55 بلا تكرار", R)
        check(st.archive_complete, "يُعلَّم الأرشيف مكتملاً عند بلوغ أول رسالة", R)

        # 4) تحديث: رسائل جديدة + تعديل رسالة حديثة؛ لا يعيد الزحف للأرشيف كله
        msgs[56] = ("وظيفة جديدة 56", "2026-11-05T00:00:00+00:00")
        msgs[57] = ("وظيفة جديدة 57", "2026-11-05T01:00:00+00:00")
        msgs[50] = ("وظيفة رقم 50 — تم تمديد الموعد", msgs[50][1], True)
        ch.calls.clear()
        st = tw.run(CH, "update", max_pages=50, delay=0)
        d = pd.read_csv(tw.TG_DIR / f"{CH}.csv", dtype=str)
        ed = pd.read_csv(tw.TG_DIR / f"{CH}_edits.csv", dtype=str)
        check(st.new_posts == 2 and len(d) == 57, "التحديث يضيف الجديد فقط (2)", R)
        check(len(ch.calls) <= tw.EDIT_RECHECK_PAGES + 1, f"التحديث لا يزحف للأرشيف ({len(ch.calls)} طلبات)", R)
        check(st.edited_detected == 1 and ed.iloc[0].previous_text == "وظيفة رقم 50", "رصد التعديل وحفظ النص السابق", R)
        check(d.set_index("source_message_id").loc["50", "raw_text"].endswith("تمديد الموعد"), "النص الحالي محدَّث", R)

        # 5) إعادة التشغيل دون تغيير: لا جديد ولا تعديلات
        st = tw.run(CH, "update", max_pages=50, delay=0)
        check(st.new_posts == 0 and st.edited_detected == 0, "إعادة التشغيل لا تكرر ولا تخترع تعديلات", R)

        # 6) فشل الشبكة يُسجَّل ولا يُحذف المحفوظ
        ch.fail_urls.add(f"{tw.BASE}/s/{CH}")
        st = tw.run(CH, "update", max_pages=5, delay=0)
        fails = pd.read_csv(tw.LOGS_DIR / f"tg_{CH}_failures.csv")
        check(st.pages_failed == 1 and "HTTP 503" in fails.reason.iloc[-1] and st.total_in_file == 57,
              "الفشل يُسجَّل والبيانات المحفوظة سليمة", R)

        # 7) قناة بلا معاينة عامة → 0 منشورات + سبب واضح
        empty = FakeChannel({})
        tw.Fetcher = fake_fetcher_factory(empty)
        st = tw.run("test_private", "all", max_pages=3, delay=0)
        fails = pd.read_csv(tw.LOGS_DIR / "tg_test_private_failures.csv")
        check(st.total_in_file == 0 and "0 posts parsed" in fails.reason.iloc[0], "قناة خاصة/مغلقة: 0 وتسجيل السبب", R)
    finally:
        tw.TG_DIR, tw.STATE_DIR, tw.LOGS_DIR, tw.Fetcher = orig
        shutil.rmtree(tmp, ignore_errors=True)
    ok = sum(x for _, x in R)
    print(f"\nالنتيجة: {ok}/{len(R)} (اختبار على HTML مصطنع — ليست نتائج جمع حقيقية)")
    raise SystemExit(0 if ok == len(R) else 1)


if __name__ == "__main__":
    main()
