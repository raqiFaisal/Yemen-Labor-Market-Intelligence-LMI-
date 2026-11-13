"""اختبار التجميع النهائي والفحوص: build → finalize → validate على بيانات مصطنعة، ثم إفساد متعمد للتأكد من أن الفحوص تكتشفه.
    python -m tests.test_final_dataset
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from core.config import ROOT
from tests.test_dataset_build import COURSE, HR, MULTI, REMOTE, SAUDI, SPAM, TENDER


def run(args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=cwd)


def check(c, label, R):
    R.append(bool(c))
    print(("✔ " if c else "✖ ") + label)


def fixture(tmp: Path):
    cols = ["source", "source_type", "source_name", "source_channel", "source_message_id", "source_url", "message_date", "raw_text",
            "text_hash", "edited", "forwarded_from", "has_photo", "has_video", "has_document", "media_only", "link_urls", "views",
            "collected_at", "last_seen_at", "collector", "parser_version"]
    rows = []
    variants = [("hr_yemen", HR), ("job_yemen", MULTI), ("forus_taiz", COURSE), ("forus_taiz", SPAM), ("hr_yemen", TENDER),
                ("TheYemenGate", REMOTE), ("jobsforyemeni", SAUDI)]
    roles = ["محاسب", "مهندس مدني", "سكرتيرة", "مندوب مبيعات", "مدرس لغة إنجليزية", "صيدلي", "سائق", "ممرضة", "مصمم جرافيك", "مبرمج"]
    govs = ["صنعاء", "عدن", "تعز", "إب", "حضرموت", "الحديدة", "مأرب", "ذمار"]
    k = 0
    for i in range(60):                       # 60 شهراً × عدة إعلانات مختلفة فعلاً
        month = pd.Timestamp("2021-01-01") + pd.DateOffset(months=i)
        for j in range(12):
            k += 1
            role, gov = roles[(i + j) % len(roles)], govs[(i * 3 + j) % len(govs)]
            text = (f"مطلوب {role} للعمل في شركة رقم {k} في {gov}\nالشروط: بكالوريوس وخبرة {1 + j % 5} سنوات\n"
                    f"إجادة Excel و English والتواصل\nالراتب: {50 + j} الف ريال\nللتواصل 77{k:07d}")
            rows.append({c: "" for c in cols} | {"source_channel": "job_yemen", "source_message_id": str(k),
                                                 "source_url": f"https://t.me/job_yemen/{k}",
                                                 "message_date": (month + pd.Timedelta(days=j)).isoformat() + "+00:00",
                                                 "raw_text": text, "media_only": "False"})
    for n, (ch, t) in enumerate(variants * 3):
        k += 1
        rows.append({c: "" for c in cols} | {"source_channel": ch, "source_message_id": str(k), "source_url": f"https://t.me/{ch}/{k}",
                                             "message_date": f"2025-{1 + n % 12:02d}-10T08:00:00+00:00", "raw_text": t + f"\n{n}",
                                             "media_only": "False"})
    pd.DataFrame(rows).to_csv(tmp / "posts_sample.csv", index=False)
    pd.DataFrame(columns=["telegraph_path", "telegraph_url", "title", "content_text"]).to_csv(tmp / "telegraph_sample.csv", index=False)


def main():
    R = []
    tmp = Path(tempfile.mkdtemp(prefix="lmi_final_test_"))
    try:
        fixture(tmp)
        b = run(["scripts.build_dataset", "--sample", str(tmp), "--out", str(tmp / "ds"), "--skills", "lexicon"])
        check(b.returncode == 0, "build_dataset" + ("" if not b.returncode else ": " + b.stderr[-300:]), R)
        f = run(["scripts.finalize_dataset", "--src", str(tmp / "ds"), "--labels", str(tmp / "ds" / "labels"), "--out", str(tmp / "F")])
        check(f.returncode == 0, "finalize_dataset" + ("" if not f.returncode else ": " + f.stderr[-300:]), R)
        F = tmp / "F"
        need = ["all_posts.csv", "all_opportunities.csv", "unique_opportunities.csv", "duplicate_pairs.csv", "classification_opportunity_type",
                "classification_geographic_scope", "ner", "skills", "timeseries", "gold_annotation", "review_queue", "dataset_statistics"]
        check(all((F / n).exists() for n in need), "كل الملفات والمجلدات المطلوبة", R)
        v = run(["scripts.validate_final_dataset", "--dir", str(F)])
        check(v.returncode == 0, "الفحوص تنجح على بيانات سليمة" + ("" if not v.returncode else ": " + v.stdout[-500:]), R)
        for n in ["data_dictionary.md", "dataset_quality_report.html", "dataset_manifest.json"]:
            check((F / n).exists() and (F / n).stat().st_size > 500, f"{n} مُنشأ", R)
        man = json.loads((F / "dataset_manifest.json").read_text(encoding="utf-8"))
        check(all("sha256" in x for x in man["files"]) and any("columns" in x for x in man["files"]), "manifest: بصمات وأعمدة وأنواع", R)
        suff = json.loads((F / "timeseries" / "lstm_sufficiency.json").read_text(encoding="utf-8"))
        check(suff["skills_ready"] > 0 and suff["lstm_split"] is not None,
              f"LSTM: سلسلة 60 شهراً تكفي ({suff['skills_ready']} مهارة، نافذة {suff['window']})", R)
        ls = pd.read_csv(F / "timeseries" / "lstm_ready" / "skill_share_series.csv")
        check(list(ls.split[-6:]) == ["test"] * 6 and list(ls.split[-9:-6]) == ["validation"] * 3, "LSTM: آخر 6 أشهر test و3 قبلها validation", R)

        # --- إفساد متعمد: يجب أن تفشل الفحوص
        G = tmp / "G"
        shutil.copytree(F, G)
        tr = pd.read_csv(G / "classification_opportunity_type" / "train.csv", dtype=str, keep_default_na=False)
        te = pd.read_csv(G / "classification_opportunity_type" / "test.csv", dtype=str, keep_default_na=False)
        pd.concat([te, tr.head(3)]).to_csv(G / "classification_opportunity_type" / "test.csv", index=False)
        lines = open(G / "ner" / "train.jsonl", encoding="utf-8").read().splitlines()
        e = json.loads(lines[0])
        e["ner_tags"][0] = "I-SKILL"            # I- بلا B-
        e["ner_tags"].append("O")               # طول غير متساوٍ
        lines[0] = json.dumps(e, ensure_ascii=False)
        open(G / "ner" / "train.jsonl", "w", encoding="utf-8").write("\n".join(lines) + "\n")
        v2 = run(["scripts.validate_final_dataset", "--dir", str(G)])
        check(v2.returncode == 1, "الفحوص تفشل عند إفساد البيانات (رمز خروج 1)", R)
        check("✖ [classification_opportunity_type] لا تداخل في group بين الأقسام" in v2.stdout, "اكتشاف تسرب train→test", R)
        check("✖ NER: صحة BIO" in v2.stdout, "اكتشاف BIO/طول غير صالح", R)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nالنتيجة: {sum(R)}/{len(R)} (بيانات مصطنعة — ليست نتائج حقيقية)")
    raise SystemExit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
