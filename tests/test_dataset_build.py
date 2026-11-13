"""اختبار بناء الـ Dataset (المرحلة 5) على منشورات مصطنعة بصيغ القنوات المرصودة — في مجلد مؤقت، بلا بيانات حقيقية.
    python -m tests.test_dataset_build
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from core.config import ROOT
from preprocessing.job_fields import extract_all
from preprocessing.opportunity import channel_boilerplate, classify_scope, classify_type, strip_boilerplate
from preprocessing.telegram_parse import parse_post, split_positions

HR = ("وظيفة رقم #12\n\n📋 المسمى الوظيفي:\nProject Officer\n\n🏢 الجهة:\nDRC\n\n📍 الموقع:\nتعز\n\n📅 تاريخ النشر:\n02 نوفمبر 2026\n\n"
      "⏰ آخر موعد للتقديم:\n15 نوفمبر 2026\n\n━━━━━━━━━━━━━━━━━━━━\n\n📋 الوصف الوظيفي: بكالوريوس في التنمية. خبرة 3 سنوات. Excel\n"
      "━━━━━━━\nقناة وظائف اليمن\nhttps://t.me/hr_yemen\n━━━━━━━\nنتمنى لكم التوفيق! ❤️")
MULTI = ("وظائف شاغرة\nيعلن جوب فايندر للتوظيف عن توفر الوظائف التالية في صنعاء\n1- محاسب\nالشروط: خبرة سنتين\nالراتب: بعد المقابلة\n"
         "2- سكرتيرة\nخبرة لا تقل عن سنة\n3- محاسب\nبكالوريوس محاسبة\nللتواصل 777123456")
COURSE = "✨ دورة احترافية في التصميم ✨\nشهادة معتمدة\n📌 المقاعد محدودة\n📞 للحجز 730123481"
TENDER = "Yemen HR | Jobs and Tenders in Yemen (RSS)\nRFQ: Supply of hygiene kits – Amran\nAmran\n05 Jan. 2023"
SPAM = "#للبيع\nسيارة هايلكس 2013 للبيع"
REMOTE = "وظيفة جديدة ✨🇾🇪 عمل عن بعد\n\nمطور Full-Stack\n\nنبحث عن مطور React و Node.js\n\nhttps://go.al-yemen.com/x"
SAUDI = "مطلوب محاسب للعمل في الرياض - السعودية\nخبرة 5 سنوات"
FOOTER = "\n#وظائف #عمل #منح #دورات\nتابعونا على واتساب"


def check(c, label, R):
    R.append(bool(c))
    print(("✔ " if c else "✖ ") + label)


def unit(R):
    f = parse_post(HR, "hr_yemen", "2026-11-02T08:00:00+00:00")
    check(f["template"] == "hr_yemen_labelled" and f["title"] == "Project Officer" and f["organization"] == "DRC"
          and f["location_text"] == "تعز", "قالب @hr_yemen: المسمى/الجهة/الموقع", R)
    check(f["published_date"] == "2026-11-02" and f["application_deadline"] == "2026-11-15",
          "تاريخ النشر والموعد النهائي من نص الإعلان (منفصلان عن تاريخ الرسالة)", R)
    t = classify_type(HR, f, "hr_yemen")
    check(t["opportunity_type"] == "JOB" and t["type_status"] == "accepted", "وظيفة بقالب = JOB مقبول", R)
    for txt, exp, lab in [(COURSE, "TRAINING", "دورة"), (TENDER, "NOT_RELEVANT", "مناقصة"), (SPAM, "NOT_RELEVANT", "بيع")]:
        ff = parse_post(txt, "forus_taiz")
        check(classify_type(txt, ff, "forus_taiz")["opportunity_type"] == exp, f"{lab} → {exp}", R)
    check(classify_type("", {}, "x")["type_status"] == "excluded", "منشور بلا نص يُستبعد (لا يُصنَّف تخميناً)", R)
    rp = "Yemen HR - Yemen's Premier Jobs & Tenders Platform\nFinance Officer\n#yemenhr\n\nAden\n\nsource\n(author: NRC)"
    check(classify_type(rp, parse_post(rp, "hr_yemen"), "hr_yemen")["opportunity_type"] == "JOB",
          "إعادة نشر YemenHR (ترويسة «Jobs & Tenders») = وظيفة لا مناقصة", R)
    ins = "وظيفة جديدة 🌏✨ عمل عن بعد\n\nMutqinai : انستراكتور كاتب محتوى\nنبحث عن مدرب لتقديم كورس كتابة المحتوى وإعداد الدورات والكورسات"
    check(classify_type(ins, parse_post(ins, "TheYemenGate"), "TheYemenGate")["opportunity_type"] == "JOB",
          "قالب «وظيفة جديدة» لمدرّب: كلمات «كورس/دورات» في الوصف لا تجعله دورة", R)
    sch = "وظيفة جديدة ✨ YemenHR\nبرنامج قادة الغد — منحة بكالوريوس ممولة بالكامل"
    check(classify_type(sch, parse_post(sch, "TheYemenGate"), "TheYemenGate")["opportunity_type"] == "SCHOLARSHIP",
          "منحة في العنوان تبقى منحة رغم القالب", R)
    warn = "⚠️ تحذير هام\nاحذروا روابط الاحتيال واستغلال حاجة المواطنين للمساعدات ومسابقات وهمية"
    check(classify_type(warn, parse_post(warn, "forus_taiz"), "forus_taiz")["opportunity_type"] == "NOT_RELEVANT",
          "تحذير من الاحتيال = غير ذي صلة", R)
    fr = parse_post(REMOTE, "TheYemenGate")
    check(classify_scope(REMOTE, fr)["geographic_scope"] == "REMOTE_GLOBAL", "عمل عن بعد → REMOTE_GLOBAL", R)
    fs = parse_post(SAUDI, "jobsforyemeni")
    check(classify_scope(SAUDI, fs)["geographic_scope"] == "OUTSIDE_YEMEN", "الرياض → OUTSIDE_YEMEN", R)
    check(classify_scope(HR, f)["governorate"] == "تعز", "المحافظة من حقل الموقع", R)
    parts = split_positions(MULTI, "job_yemen")
    check([p["role_line"] for p in parts] == ["محاسب", "سكرتيرة", "محاسب"], "تقسيم منشور بثلاث وظائف مرقّمة", R)
    check(split_positions(HR, "hr_yemen") == [], "لا تقسيم لقالب @hr_yemen", R)
    ex = extract_all("بكالوريوس في المحاسبة. خبرة 3 إلى 5 سنوات. دوام كامل")
    check(ex["education_min"] == "Bachelor" and ex["experience_min_years"] == 3 and ex["experience_max_years"] == 5
          and "full_time" in ex["employment_type"], "المؤهل والخبرة ونوع الدوام", R)
    check(extract_all("خبرة 6 أشهر")["experience_min_years"] is None, "«6 أشهر» ليست 6 سنوات", R)
    empty = extract_all("مطلوب عامل")
    check(empty["education"] == "" and empty["experience_min_years"] is None, "لا قيم مخترعة عند الغياب", R)
    from preprocessing.dedup import find_duplicates
    D = pd.Timestamp("2020-01-01")
    chain = pd.DataFrame([(f"tg:c:{k}", "Project Officer", "DRC", "تعز", D + pd.Timedelta(days=20 * k), f"نص {k}", [])
                          for k in range(30)], columns="job_id title organization governorate date text link_keys".split())
    a, _ = find_duplicates(chain)
    span = chain.merge(a, on="job_id").groupby("canonical_job_id").date.agg(lambda x: (x.max() - x.min()).days).max()
    check(a.canonical_job_id.nunique() > 1 and span <= 90,
          f"إعلان يتكرر كل 20 يوماً لسنتين لا يصبح إعلاناً واحداً (أقصى مدى مجموعة {span} يوماً)", R)
    posts = [COURSE + FOOTER] * 30 + ["مطلوب محاسب" + FOOTER]
    bp = channel_boilerplate(posts, min_count=5)
    clean = strip_boilerplate("مطلوب محاسب" + FOOTER, bp)
    check("#منح" not in clean and "مطلوب محاسب" in clean, "إزالة تذييل القناة المتكرر (لا يعطي إشارة «منح» زائفة)", R)


def end_to_end(R):
    tmp = Path(tempfile.mkdtemp(prefix="lmi_ds_test_"))
    try:
        rows = []
        def add(ch, mid, date, text, links=""):
            rows.append({"source": f"telegram:{ch}", "source_type": "telegram", "source_name": ch, "source_channel": ch,
                         "source_message_id": str(mid), "source_url": f"https://t.me/{ch}/{mid}", "message_date": date,
                         "raw_text": text, "text_hash": "", "edited": "False", "forwarded_from": "", "has_photo": "False",
                         "has_video": "False", "has_document": "False", "media_only": str(not text), "link_urls": links,
                         "views": "", "collected_at": "2026-11-05T00:00:00+00:00", "last_seen_at": "", "collector": "test",
                         "parser_version": "test"})
        add("hr_yemen", 1, "2026-11-02T08:00:00+00:00", HR, "https://telegra.ph/Project-Officer-11-02")
        add("TheYemenGate", 2, "2026-11-03T08:00:00+00:00",
            "وظيفة جديدة ✨🇾🇪 YemenHR\nProject Officer\nTaiz\n\nPosted: 02 Nov, 26\nCompany: DRC\nLocation: Taiz")
        add("job_yemen", 3, "2016-01-10T08:00:00+00:00", MULTI)
        add("job_yemen", 4, "2016-01-12T08:00:00+00:00", MULTI)               # إعادة نشر نفس القائمة
        add("forus_taiz", 5, "2026-10-01T08:00:00+00:00", COURSE)
        add("forus_taiz", 6, "2026-10-02T08:00:00+00:00", SPAM)
        add("forus_taiz", 7, "2026-10-03T08:00:00+00:00", "")
        add("TheYemenGate", 8, "2026-10-04T08:00:00+00:00", REMOTE)
        pd.DataFrame(rows).to_csv(tmp / "posts_sample.csv", index=False)
        pd.DataFrame([{"telegraph_path": "Project-Officer-11-02", "telegraph_url": "https://telegra.ph/Project-Officer-11-02",
                       "title": "Project Officer", "content_text": "Job Title: Project Officer\nRequirements:\n- Bachelor's degree in "
                       "Development Studies\n- At least 3 years of experience\n- Excel and report writing", "first_message_id": "1"}]) \
            .to_csv(tmp / "telegraph_sample.csv", index=False)
        out = tmp / "out"
        r = subprocess.run([sys.executable, "-m", "scripts.build_dataset", "--sample", str(tmp), "--out", str(out),
                            "--skills", "lexicon"], capture_output=True, text=True, cwd=ROOT)
        check(r.returncode == 0, "build_dataset يعمل" + ("" if r.returncode == 0 else ": " + r.stderr[-400:]), R)
        if r.returncode:
            return
        o = pd.read_pickle(out / "opportunities.pkl")
        pa = pd.read_csv(out / "posts_all.csv", dtype=str, keep_default_na=False)
        hr = o[o.job_id == "tg:hr_yemen:1"].iloc[0]
        check(hr.telegraph_url.endswith("Project-Officer-11-02") and "Development Studies" in hr.description,
              "دمج صفحة telegra.ph مع منشورها (النص الكامل في description)", R)
        check(hr.raw_text == HR, "raw_text محفوظ حرفياً", R)
        check(hr.message_date.startswith("2026-11-02") and hr.published_date == "2026-11-02" and hr.application_deadline == "2026-11-15",
              "تاريخ الرسالة وتاريخ الإعلان والموعد النهائي حقول منفصلة", R)
        gate = o[o.job_id == "tg:TheYemenGate:2"].iloc[0]
        check(gate.canonical_job_id == hr.canonical_job_id and {hr.dup_status, gate.dup_status} == {"duplicate"},
              "نفس الوظيفة في قناتين → نفس canonical_job_id", R)
        check(o[o.canonical_job_id == hr.canonical_job_id].is_canonical.sum() == 1, "ممثل واحد لكل مجموعة تكرار", R)
        kids3, kids4 = o[o.post_id == "tg:job_yemen:3"], o[o.post_id == "tg:job_yemen:4"]
        check(len(kids3) == 3 and len(kids4) == 3, "المنشور متعدد الوظائف → 3 سجلات", R)
        k3 = kids3.set_index("position_index").canonical_job_id
        check(k3[1] != k3[3], "وظيفتان بنفس المسمى في نفس المنشور ليستا تكراراً", R)
        check(set(kids3.canonical_job_id) == set(kids4.canonical_job_id), "إعادة نشر القائمة بعد يومين → مكررة", R)
        check(len(o) == 10 and int((pa.excluded_reason != "").sum()) == 2,
              f"الصفوف: 10 فرص (2 + 3×2 + دورة + عن بعد)؛ منشوران مستبعدان (بيع + بلا نص) — فعلي {len(o)}", R)
        check(set(o.groupby("canonical_job_id").split_random.nunique()) == {1}, "لا تنقسم مجموعة تكرار بين train/val/test", R)
        ner = [json.loads(l) for sp in ("train", "val", "test") for l in open(out / "ner" / f"silver_{sp}.jsonl", encoding="utf-8")]
        ok_spans = all(e["text"][s["start"]:s["end"]] for e in ner for s in e["spans"]) and all(
            len(e["tokens"]) == len(e["ner_tags"]) for e in ner)
        check(ner and ok_spans, f"NER: {len(ner)} أمثلة، المواضع داخل النص والوسوم بطول الرموز", R)
        rep = json.loads((out / "reports" / "dataset_report.json").read_text(encoding="utf-8"))
        check(rep["posts_total"] == 8 and "SILVER" in rep["label_quality"], "التقرير: أعداد فعلية + تصريح بأن التسميات Silver", R)
        exp = pd.read_csv(out / "telegram_jobs_for_training.csv")
        check(len(exp) == exp.source_job_id.nunique() and "TRAINING" not in set(o[o.job_id.isin(exp.source_job_id)].opportunity_type),
              "ملف التدريب: وظائف فريدة فقط (بلا دورات)", R)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    R = []
    unit(R)
    end_to_end(R)
    print(f"\nالنتيجة: {sum(R)}/{len(R)} (بيانات اختبار مصطنعة — ليست نتائج حقيقية)")
    raise SystemExit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
