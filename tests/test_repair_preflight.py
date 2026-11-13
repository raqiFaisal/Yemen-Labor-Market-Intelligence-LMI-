import json
from pathlib import Path
from scripts.verify_delivery import verify
from src.inference.predictors import word_windows, fix_bio, ModelNotAvailable, TextClassifier

ROOT = Path(__file__).resolve().parents[1]

def test_preflight_existing_dataset():
    report = verify(ROOT/'FINAL_TRAINING_DATASET')
    assert not report['failures'], report['failures']
    assert report['source_provenance']['unique_opportunities'] > 20000
    assert report['ner']['requested_but_absent'] == ['DATE','EDUCATION']

def test_word_chunking_no_tokenizer():
    assert word_windows(list('abcdefg'),None,3,None) == [(0,3),(3,6),(6,7)]
    assert word_windows([],None,3,None) == []

def test_repair_invalid_bio():
    assert fix_bio(['I-SKILL','I-SKILL','I-LOCATION']) == ['B-SKILL','I-SKILL','B-LOCATION']

def test_no_fake_model_fallback(tmp_path):
    try:
        TextClassifier(tmp_path)
    except ModelNotAvailable:
        pass
    else:
        raise AssertionError('Missing model silently accepted')
