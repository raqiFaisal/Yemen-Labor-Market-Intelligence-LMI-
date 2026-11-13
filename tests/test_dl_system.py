"""اختبارات نظام التعلم العميق: القراءة، المقاييس، المحاذاة، تحميل النماذج وغيابها، محرك التكامل، فجوة المهارات، الصفحات.

كل البيانات هنا اصطناعية تُنشأ داخل مجلد مؤقت للاختبار فقط (fixture) — لا تمثل السوق ولا تُستخدم خارج الاختبارات.
النموذج المستخدم BERT صغير عشوائي (بلا إنترنت) للتحقق من مسارات الكود، لا لقياس الأداء.

    pytest -q tests/test_dl_system.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLES = [("محاسب", "محاسبة", ["excel", "accounting"]), ("مهندس مدني", "هندسة مدنية", ["autocad", "english"]),
          ("مطور ويب", "علوم حاسوب", ["javascript", "react", "english"]), ("ممرض", "تمريض", ["english"]),
          ("مسؤول مبيعات", "تسويق", ["sales", "communication"])]
GOVS = ["صنعاء", "عدن", "تعز", "حضرموت"]
CATALOG = [("excel", "إكسل", "Microsoft Excel", "technical"), ("accounting", "المحاسبة", "Accounting", "professional"),
           ("autocad", "أوتوكاد", "AutoCAD", "technical"), ("english", "اللغة الإنجليزية", "English Language", "soft"),
           ("javascript", "جافاسكربت", "JavaScript", "technical"), ("react", "رياكت", "React", "technical"),
           ("sales", "المبيعات", "Sales", "professional"), ("communication", "التواصل", "Communication", "soft")]
TAGS = ["O"] + [f"{p}-{e}" for e in ["JOB_TITLE", "ORGANIZATION", "SKILL", "DEGREE", "MAJOR", "EXPERIENCE", "LOCATION",
                                     "SALARY", "DEADLINE"] for p in "BI"]


def _ner_record(i, title, gov, skill_ar):
    toks = ["مطلوب"] + title.split() + ["في"] + [gov] + ["إجادة"] + [skill_ar.split()[0]]
    tags = ["O", "B-JOB_TITLE"] + ["I-JOB_TITLE"] * (len(title.split()) - 1) + ["O", "B-LOCATION", "O", "B-SKILL"]
    return {"id": f"j{i}", "text": " ".join(toks), "tokens": toks, "ner_tags": tags}


@pytest.fixture(scope="session")
def final_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("FINAL_TEST_ONLY")
    rows, js = [], []
    months = pd.period_range("2023-01", periods=30, freq="M")
    i = 0
    for mi, m in enumerate(months):
        for k in range(12):
            t, spec, sk = TITLES[(mi + k) % len(TITLES)]
            g = GOVS[k % len(GOVS)]
            typ = "INTERNSHIP" if k == 11 else "JOB"
            jid = f"j{i}"
            rows.append({"job_id": jid, "post_id": f"tg:x:{i}", "opportunity_type": typ, "geographic_scope": "SINGLE_GOVERNORATE",
                         "job_title": t, "organization": f"شركة {k % 3}", "organization_canonical": f"شركة {k % 3}",
                         "governorate": g, "specializations": spec, "education": "بكالوريوس",
                         "experience_min_years": str(k % 4), "posting_date": f"{m}-{(k % 27) + 1:02d}",
                         "application_deadline": "", "source_channel": "x", "source_url": f"https://t.me/x/{i}",
                         "raw_text": f"مطلوب {t} في {g}", "text_clean": f"مطلوب {t} في {g}"})
            for s in sk:
                js.append({"job_id": jid, "skill_id": s, "decision": "accepted",
                           **dict(zip(["name_ar", "name_en", "skill_type"], [c[1:] for c in CATALOG if c[0] == s][0]))})
            i += 1
    u = pd.DataFrame(rows)
    u.to_csv(d / "unique_opportunities.csv", index=False)
    (d / "skills").mkdir()
    pd.DataFrame(js).to_csv(d / "skills" / "job_skills.csv", index=False)
    pd.DataFrame(CATALOG, columns=["skill_id", "name_ar", "name_en", "skill_type"]).to_csv(d / "skills" / "skill_catalog.csv", index=False)
    # التصنيف
    for task, labels, col in [("classification_opportunity_type", ["JOB", "INTERNSHIP"], "opportunity_type"),
                              ("classification_geographic_scope", ["SINGLE_GOVERNORATE", "REMOTE"], "geographic_scope")]:
        f = d / task
        f.mkdir()
        lm = {"label2id": {l: n for n, l in enumerate(labels)}, "id2label": {str(n): l for n, l in enumerate(labels)}}
        (f / "label_map.json").write_text(json.dumps(lm), encoding="utf-8")
        dd = u.assign(id=u.job_id, text=u.raw_text, label=u[col])
        if col == "geographic_scope":
            dd.loc[dd.index % 5 == 0, "label"] = "REMOTE"
            dd.loc[dd.index % 5 == 0, "text"] = dd.text + " عن بعد"
        dd["label_id"] = dd.label.map(lm["label2id"])
        for sp, part in [("train", dd.iloc[:200]), ("validation", dd.iloc[200:280]), ("test", dd.iloc[280:])]:
            part[["id", "text", "label", "label_id"]].to_csv(f / f"{sp}.csv", index=False)
    # NER
    n = d / "ner"
    n.mkdir()
    (n / "label_list.json").write_text(json.dumps({"bio_tags": TAGS}), encoding="utf-8")
    recs = [_ner_record(j, r["job_title"], r["governorate"], "إكسل") for j, r in enumerate(rows)]
    for sp, part in [("train", recs[:200]), ("validation", recs[200:280]), ("test", recs[280:])]:
        with open(n / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    return d


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory, final_dir) -> Path:
    """BERT صغير عشوائي + Tokenizer WordPiece مدرّب على نصوص الـ fixture (بدون إنترنت)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    d = tmp_path_factory.mktemp("tiny_bert")
    texts = pd.read_csv(final_dir / "unique_opportunities.csv").raw_text.tolist() + ["عن بعد إجادة إكسل"]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = BertTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                            mask_token="[MASK]")
    tok.save_pretrained(d)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=256)
    BertForMaskedLM(cfg).save_pretrained(d)
    return d


def _run(mod, *args, env):
    r = subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="session")
def trained(tmp_path_factory, final_dir, tiny_base):
    """تدريب Epoch واحد لكل مهمة على النموذج الصغير عبر نفس سكربتات التدريب والتقييم الحقيقية."""
    models = tmp_path_factory.mktemp("models")
    reports = tmp_path_factory.mktemp("reports")
    env = {**os.environ, "LMI_FINAL_DIR": str(final_dir), "LMI_MODELS_DIR": str(models), "LMI_REPORTS_DIR": str(reports)}
    for task in ("opportunity_type", "geographic_scope"):
        _run("src.training.train_classifier", "--task", task, "--model", str(tiny_base), "--epochs", "1", "--lr", "1e-3",
             "--max_length", "32", env=env)
        _run("src.evaluation.evaluate_classifier", "--task", task, env=env)
    _run("src.training.train_ner", "--model", str(tiny_base), "--epochs", "1", "--lr", "1e-3", env=env)
    _run("src.evaluation.evaluate_ner", env=env)
    return {"models": models, "reports": reports, "env": env}


# ───────────── القراءة والمقاييس ─────────────
def test_loaders_validate(final_dir):
    from src.data.loaders import load_classification, load_ner
    data, lm = load_classification("opportunity_type", final_dir)
    assert set(data) == {"train", "validation", "test"} and set(data["train"].label) <= set(lm["label2id"])
    ner, tags = load_ner(final_dir)
    assert all(len(r["tokens"]) == len(r["ner_tags"]) for r in ner["train"])
    assert not set(r["id"] for r in ner["train"]) & set(r["id"] for r in ner["test"])


def test_loader_rejects_unknown_label(tmp_path, final_dir):
    import shutil
    from src.data.loaders import load_classification
    d = tmp_path / "F"
    shutil.copytree(final_dir / "classification_opportunity_type", d / "classification_opportunity_type")
    f = d / "classification_opportunity_type" / "test.csv"
    t = pd.read_csv(f)
    t.loc[0, "label"] = "BOGUS"
    t.to_csv(f, index=False)
    with pytest.raises(ValueError):
        load_classification("opportunity_type", d)


def test_entity_metrics_strict_and_json_safe():
    from src.evaluation.metrics import entity_metrics
    true = [["B-SKILL", "I-SKILL", "O", "B-LOCATION"]]
    pred = [["B-SKILL", "O", "O", "B-LOCATION"]]                  # حدود المهارة خاطئة → خطأ صارم
    m = entity_metrics(true, pred)
    assert m["entity_precision"] == 0.5 and m["entity_recall"] == 0.5
    json.dumps(m)                                                  # أنواع Python أصلية (لا np.float64)
    assert all(type(x["f1"]) is float for x in m["per_entity"])


def test_forecast_metrics_mape_floor():
    from src.evaluation.metrics import forecast_metrics
    m = forecast_metrics([0.5, 0.001, np.nan], [0.4, 0.5, 0.3])
    assert m["n"] == 2 and m["mape_n"] == 1 and m["mape"] == 20.0


def test_fix_bio():
    from src.inference.predictors import fix_bio
    assert fix_bio(["I-SKILL", "I-SKILL", "B-LOCATION", "I-SKILL"]) == ["B-SKILL", "I-SKILL", "B-LOCATION", "B-SKILL"]


def test_ner_alignment_masks_subwords(tiny_base):
    from transformers import AutoTokenizer
    from src.training.train_ner import align
    tok = AutoTokenizer.from_pretrained(tiny_base)
    chunks = [(["مطلوب", "محاسبمحاسبمحاسب", "في", "عدن"], ["O", "B-JOB_TITLE", "O", "B-LOCATION"])]
    l2i = {t: i for i, t in enumerate(TAGS)}
    enc, lost = align(tok, chunks, l2i, 64)
    labs, wids = enc["labels"][0], enc.word_ids(0)
    assert labs[0] == -100 and labs[-1] == -100                   # [CLS] / [SEP]
    firsts = [i for i, w in enumerate(wids) if w is not None and (i == 0 or wids[i - 1] != w)]
    assert [labs[i] for i in firsts] == [l2i[t] for t in chunks[0][1]]
    assert all(labs[i] == -100 for i, w in enumerate(wids) if w is not None and i not in firsts)
    assert lost == 0


# ───────────── النماذج ─────────────
def test_missing_model_raises(tmp_path):
    from src.inference.predictors import ModelNotAvailable, NERTagger, TextClassifier
    with pytest.raises(ModelNotAvailable):
        TextClassifier(tmp_path / "nope")
    with pytest.raises(ModelNotAvailable):
        NERTagger(tmp_path / "nope")


def test_training_artifacts_and_no_test_leak(trained):
    for task in ("opportunity_type", "geographic_scope"):
        d = trained["models"] / "classification" / task
        for f in ("config.json", "label_map.json", "training_config.json", "training_history.csv", "best_validation_metrics.json"):
            assert (d / f).exists(), f
        cfg = json.loads((d / "training_config.json").read_text(encoding="utf-8"))
        assert cfg.get("test_used_during_training") is False
        ev = json.loads((trained["reports"] / "evaluation" / f"classification_{task}" / "metrics.json").read_text(encoding="utf-8"))
        assert ev["split"] == "test" and 0 <= ev["macro_f1"] <= 1
    ner = json.loads((trained["reports"] / "evaluation" / "ner" / "metrics.json").read_text(encoding="utf-8"))
    assert ner["split"] == "test" and "per_entity" in ner


def test_predictors_load_and_predict(trained):
    from src.inference.predictors import NERTagger, TextClassifier
    clf = TextClassifier(trained["models"] / "classification" / "opportunity_type")
    p = clf.predict("مطلوب محاسب في عدن")
    assert p["label"] in clf.labels and abs(sum(t["probability"] for t in clf.predict("x", top_k=2)["top"]) - 1) < 1.01
    assert np.allclose(clf.predict_proba(["a", "b"]).sum(1), 1, atol=1e-4)
    ner = NERTagger(trained["models"] / "ner")
    r = ner.extract("مطلوب محاسب في عدن إجادة إكسل " * 80)   # نص طويل → تقطيع إلى مقاطع
    assert len(r["tags"]) == len(r["tokens"]) and all(t in ner.labels for t in r["tags"])


# ───────────── محرك التكامل ─────────────
@pytest.fixture()
def engine(final_dir, trained):
    from src.integration.engine import Engine
    return Engine(final_dir, trained["models"], trained["reports"])


def test_market_indicators(engine):
    m = engine.market
    o = m.overview()
    assert o["unique_jobs_internships"] == 360 and o["governorates_covered"] == 4
    ts = m.top_skills(3)
    assert ts.iloc[0].skill_id == "english" and ts.jobs.is_monotonic_decreasing
    from src.integration.canon import canon_spec
    assert set(m.top_multi("specializations", 10).specializations) == {canon_spec(t[1]) for t in TITLES}
    r = m.spec_skill(min_jobs=5)
    assert r["share"].loc["المحاسبة", "إكسل"] == 1.0 and r["share"].loc["تمريض", "إكسل"] == 0.0
    assert len(m.search("محاسب", governorate="عدن")) > 0
    assert m.search("كلمة_غير_موجودة").empty


def test_analyze_ad_uses_models(engine):
    r = engine.analyze_ad("مطلوب محاسب في عدن إجادة إكسل وخبرة سنتين")
    assert r["all_models_available"]
    assert {"opportunity_type", "geographic_scope", "ner"} == set(r["components"])
    assert r["components"]["opportunity_type"]["status"] == "ok"


def test_analyze_ad_missing_models(final_dir, tmp_path):
    from src.integration.engine import Engine
    e = Engine(final_dir, tmp_path / "no_models", tmp_path / "no_reports")
    r = e.analyze_ad("مطلوب محاسب في عدن إجادة إكسل")
    assert not r["all_models_available"]
    assert all(c["status"] == "model_not_available" for c in r["components"].values())
    assert e.forecasts.summary() is None and e.forecasts.family("skills")["status"] == "not_available"


@pytest.mark.parametrize("bad", ["", "   ", "قصير"])
def test_analyze_ad_invalid_input(engine, bad):
    with pytest.raises(ValueError):
        engine.analyze_ad(bad)


def test_skill_gap(engine):
    g = engine.skill_gap(["Excel", "xyz-unknown"], target_specialization="المحاسبة")
    assert g.n_target_jobs == 72
    assert [x["skill_id"] for x in g.matched] == ["excel"] and [x["skill_id"] for x in g.missing] == ["accounting"]
    assert g.match_pct == 50.0 and g.unrecognized == ["xyz-unknown"]
    assert g.nearest_jobs.coverage.iloc[0] == 50.0
    assert g.recommendations[0]["skill"] == "المحاسبة"
    full = engine.skill_gap(["إكسل", "المحاسبة"], target_specialization="المحاسبة")
    assert full.match_pct == 100.0 and not full.missing


def test_skill_gap_edge_cases(engine):
    with pytest.raises(ValueError):
        engine.skill_gap([])
    g = engine.skill_gap(["excel"], target_title="رائد فضاء")
    assert g.n_target_jobs == 0 and g.note and g.match_pct == 0.0


def test_series_sufficiency(final_dir):
    from src.forecasting.series import build_family
    r = build_family("skills", final_dir)                       # 30 شهراً × 12 إعلاناً ≥ الحدود
    assert r["status"] == "ok" and r["months"] == 30 and "english" in r["ready"]
    assert r["series"].notna().all().all()


# ───────────── الصفحات ─────────────
PAGES = ["dashboard", "market", "forecast", "search", "analyze", "gap", "models", "about"]


def _page_env(monkeypatch, data, models, reports):
    monkeypatch.setenv("LMI_FINAL_DIR", str(data))
    monkeypatch.setenv("LMI_MODELS_DIR", str(models))
    monkeypatch.setenv("LMI_REPORTS_DIR", str(reports))
    for m in [k for k in sys.modules if k == "common" or k.startswith("src")]:
        del sys.modules[m]
    sys.path.insert(0, str(ROOT / "app_dl"))


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_data(page, monkeypatch, final_dir, trained):
    from streamlit.testing.v1 import AppTest
    _page_env(monkeypatch, final_dir, trained["models"], trained["reports"])
    at = AppTest.from_file(str(ROOT / "app_dl" / "views" / f"{page}.py"), default_timeout=120).run()
    assert not at.exception, at.exception[0].value


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_data(page, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    _page_env(monkeypatch, tmp_path / "d", tmp_path / "m", tmp_path / "r")
    at = AppTest.from_file(str(ROOT / "app_dl" / "views" / f"{page}.py"), default_timeout=60).run()
    assert not at.exception, at.exception[0].value
    if page not in ("about",):
        assert len(at.error) + len(at.warning) >= 1            # رسالة غياب صريحة بدل بيانات بديلة


def test_analyze_page_flow(monkeypatch, final_dir, trained):
    from streamlit.testing.v1 import AppTest
    _page_env(monkeypatch, final_dir, trained["models"], trained["reports"])
    at = AppTest.from_file(str(ROOT / "app_dl" / "views" / "analyze.py"), default_timeout=120).run()
    at.button[0].click().run()
    assert at.warning                                          # نص فارغ
    at.text_area[0].input("مطلوب محاسب في عدن إجادة إكسل").run()
    at.button[0].click().run()
    assert not at.exception and len(at.metric) == 2


def test_gap_page_flow(monkeypatch, final_dir, trained):
    from streamlit.testing.v1 import AppTest
    _page_env(monkeypatch, final_dir, trained["models"], trained["reports"])
    at = AppTest.from_file(str(ROOT / "app_dl" / "views" / "gap.py"), default_timeout=120).run()
    at.multiselect[0].set_value(["excel"]).run()
    at.button[0].click().run()
    assert not at.exception and len(at.dataframe) == 1


def test_spec_canonicalization():
    from src.integration.canon import canon_specs
    assert canon_specs("محاسبة;Accounting;المحاسبة;التسويق") == "المحاسبة;التسويق"
    assert canon_specs("ادارة اعمال") == "إدارة الأعمال" and canon_specs("") == ""
    assert canon_specs("تخصص جديد") == "تخصص جديد"          # الصيغ غير المعروفة لا تتغير
