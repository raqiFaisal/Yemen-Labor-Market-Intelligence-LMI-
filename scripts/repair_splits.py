"""Conservatively unify train/validation/test across classification and NER.

No opportunity or sentence is removed. The union-find groups by source ID,
canonical opportunity group, and strong normalized text fingerprints. Removing
numbers is used only to force *same split*, not to deduplicate opportunities.

Run once (or idempotently) from the project root:
  python -m scripts.repair_splits --apply
Original uploads and split snapshots are retained in reports/repair/original_splits.zip.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from src.common import DATA_DIR, REPORTS_DIR
from src.data.audit import _norm

SPLITS = ("train", "validation", "test")
TASKS = ("classification_opportunity_type", "classification_geographic_scope")


class DSU:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.sz: dict[str, int] = {}

    def find(self, s):
        if s not in self.parent:
            self.parent[s], self.sz[s] = s, 1
            return s
        x = s
        while x != self.parent[x]:
            x = self.parent[x]
        while s != x:
            p = self.parent[s]
            self.parent[s] = x
            s = p
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.parent[b] = a
        self.sz[a] += self.sz[b]


def keys(identifier: str, group: str, txt: str) -> list[str]:
    out = []
    if identifier:
        out.append("id:" + identifier)
    if group:
        out.append("canon:" + group)
    # A long matching ad after numbers are removed is a conservative split
    # signal: two distinct openings may be placed together, never deleted.
    cleaned = _norm(txt, digits=False)
    if len(cleaned) >= 25:
        out.append("txt_no_digits:" + hashlib.sha256(cleaned.encode()).hexdigest())
    else:
        cleaned = _norm(txt)
        if len(cleaned) >= 30:
            out.append("txt_exact:" + hashlib.sha256(cleaned.encode()).hexdigest())
    return out


def build(data_dir: Path) -> tuple[dict[str, str], dict]:
    dsu = DSU()
    refs: dict[str, tuple[str, str, str]] = {}
    votes: dict[str, Counter] = defaultdict(Counter)
    # Assign a stable row key; multiple task appearances of one ad MUST be in
    # the same split. Count all appearances for minimal perturbation.
    for task in TASKS:
        count = 0
        for split in SPLITS:
            df = pd.read_csv(data_dir / task / f"{split}.csv", dtype=str, keep_default_na=False)
            for r in df.itertuples(index=False):
                rid = f"cls:{task}:{split}:{count}"
                count += 1
                kk = keys(str(r.id), str(r.group), str(r.text))
                if not kk:
                    raise ValueError(f"Cannot group row {rid}: missing ID, group and usable text")
                for k in kk[1:]:
                    dsu.union(kk[0], k)
                refs[rid] = (kk[0], split, str(r.id))
    for split in SPLITS:
        with (data_dir / "ner" / f"{split}.jsonl").open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                rid = f"ner:{split}:{i}"
                kk = keys(str(r["id"]), str(r.get("canonical_job_id", "")), str(r.get("text") or " ".join(r["tokens"])))
                for k in kk[1:]:
                    dsu.union(kk[0], k)
                refs[rid] = (kk[0], split, str(r["id"]))
    # Need second pass after all unions; earlier roots may have changed.
    grouped_rows = defaultdict(list)
    for rid, (key, old, identifier) in refs.items():
        grouped_rows[dsu.find(key)].append((rid, old, identifier))
    plan: dict[str, str] = {}
    changed = Counter()
    cross_group = 0
    id_assignment = {}
    for rr in grouped_rows.values():
        votes = Counter(old for _, old, _ in rr)
        if len(votes) > 1:
            cross_group += 1
        # Favor majority, then keep training stable, then validation, then test.
        choice = sorted(SPLITS, key=lambda s: (-votes[s], SPLITS.index(s)))[0]
        for rid, old, identifier in rr:
            plan[rid] = choice
            id_assignment[identifier] = choice
            if old != choice:
                changed[f"{old}->{choice}"] += 1
    return plan, {
        "original_records": len(refs),
        "connected_split_groups": len(grouped_rows),
        "groups_with_conflicting_original_splits": cross_group,
        "moved_rows": sum(changed.values()),
        "moves": dict(changed),
        "id_assignment": id_assignment,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Group by id, canonical group, and normalized near-identical long text. Same split; never remove or merge distinct records.",
    }


def apply(data_dir: Path, report_dir: Path, plan: dict, report: dict):
    backup = report_dir / "original_splits.zip"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        with ZipFile(backup, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
            for task in TASKS:
                for split in SPLITS:
                    p = data_dir / task / f"{split}.csv"
                    z.write(p, arcname=str(p.relative_to(data_dir)))
            p = data_dir / 'classification_opportunity_type' / 'train_balanced.csv'
            if p.exists():
                z.write(p, arcname=str(p.relative_to(data_dir)))
            for split in SPLITS:
                p = data_dir / 'ner' / f"{split}.jsonl"
                z.write(p, arcname=str(p.relative_to(data_dir)))
    # Build in memory per small-ish task, and use temporary files for atomically
    # replacing outputs only after all target files have been built.
    outputs = {}
    for task in TASKS:
        arrays = defaultdict(list)
        idx = 0
        for split in SPLITS:
            df = pd.read_csv(data_dir / task / f"{split}.csv", dtype=str, keep_default_na=False)
            destinations = [plan[f"cls:{task}:{split}:{i}"] for i in range(idx, idx + len(df))]
            idx += len(df)
            for sp in SPLITS:
                arrays[sp].append(df.loc[[a == sp for a in destinations]])
        for sp in SPLITS:
            outputs[data_dir / task / f"{sp}.csv"] = pd.concat(arrays[sp], ignore_index=True).to_csv(index=False)
        if task == 'classification_opportunity_type':
            # Original train_balanced.csv was a copy of train: it must be rebuilt
            # so no moved validation/test examples can leak into model training.
            outputs[data_dir / task / 'train_balanced.csv'] = pd.concat(arrays['train'], ignore_index=True).to_csv(index=False)
    ner_rows = defaultdict(list)
    for split in SPLITS:
        for i, line in enumerate((data_dir / 'ner' / f"{split}.jsonl").read_text(encoding='utf8').splitlines()):
            if line.strip():
                ner_rows[plan[f'ner:{split}:{i}']].append(line)
    for sp in SPLITS:
        outputs[data_dir / 'ner' / f'{sp}.jsonl'] = ''.join(line + '\n' for line in ner_rows[sp])
    with tempfile.TemporaryDirectory(prefix='lmi_splits_', dir=str(report_dir)) as tmp:
        staged = []
        for idx, (target, contents) in enumerate(outputs.items()):
            temp = Path(tmp) / str(idx)
            temp.write_text(contents, encoding='utf-8-sig' if target.suffix == '.csv' else 'utf8')
            staged.append((temp, target))
        for temp, target in staged:
            temp.replace(target)
    # Keep source metadata immutable. Source split_random is original and may
    # disagree with the verified learning split; publish map explicitly.
    mapping = pd.DataFrame(sorted(report['id_assignment'].items()), columns=['id', 'verified_split'])
    mapping.to_csv(report_dir / 'verified_split_by_id.csv', index=False, encoding='utf8')
    public_report = {k: v for k, v in report.items() if k != 'id_assignment'}
    public_report['backup'] = str(backup.relative_to(report_dir.parent.parent))
    (report_dir / 'split_repair.json').write_text(json.dumps(public_report, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=DATA_DIR)
    ap.add_argument('--report-dir', type=Path, default=REPORTS_DIR / 'repair')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    plan, report = build(args.data_dir)
    print(json.dumps({k:v for k,v in report.items() if k!='id_assignment'}, ensure_ascii=False, indent=2))
    if args.apply:
        apply(args.data_dir, args.report_dir, plan, report)
        print('Applied; backups saved and no records removed.')


if __name__ == '__main__':
    main()
