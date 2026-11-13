"""Prepare human review queues WITHOUT fabricating any Gold annotations.

Never edits legacy gold files. Silver spans are only suggestions and are
renamed to match ORGANIZATION (not ORG) for display. Human-entered annotations
must have explicit completed status and reviewer ID before evaluation.
"""
import csv
import json
from pathlib import Path
from src.common import DATA_DIR

ENTITY_SET = ('JOB_TITLE','SKILL','EDUCATION','DEGREE','MAJOR','EXPERIENCE',
              'LOCATION','ORGANIZATION','SALARY','DATE')


def main(data_dir=DATA_DIR):
    folder=Path(data_dir)/'gold_annotation'
    src=folder/'ner_gold_to_annotate.jsonl';out=folder/'ner_gold_review_v2.jsonl'
    n=0
    with src.open(encoding='utf8') as fi,out.open('w',encoding='utf8') as fo:
        for line in fi:
            if not line.strip():continue
            x=json.loads(line)
            if x.get('gold_spans') is not None:
                raise ValueError('Legacy file contains manual spans: do not overwrite or transform them automatically')
            # ORG is a silver-only synonym; DEADLINE indicates a date mention
            # restricted to the application deadline, not an arbitrary date.
            hints=[]
            for a in x.get('silver_spans',[]):
                a=a.copy()
                a['label']={'ORG':'ORGANIZATION','DEADLINE':'DATE'}.get(a['label'],a['label'])
                hints.append(a)
            y={'id':x['id'],'text':x['text'],'suggested_silver_spans':hints,
               'gold_spans':None,'review_status':'pending','annotator':'','reviewed_at':'',
               'second_annotator':'','adjudicator':'','disagreement_note':''}
            fo.write(json.dumps(y,ensure_ascii=False)+'\n')
            n+=1
    s=folder/'opportunity_gold_to_annotate.csv';dst=folder/'opportunity_gold_review_v2.csv'
    with s.open(encoding='utf-8-sig',newline='') as fi,dst.open('w',encoding='utf-8-sig',newline='') as fo:
        r=csv.DictReader(fi)
        fields=r.fieldnames+['review_status','reviewed_at','second_annotator','adjudicator','adjudication_note']
        w=csv.DictWriter(fo,fieldnames=fields);w.writeheader()
        m=0
        for row in r:
            if any(row.get(k,'').strip() for k in ['gold_opportunity_type','gold_scope_detail','annotator']):
                raise ValueError('Legacy classification file already has annotations; manual migration required')
            row.update(review_status='pending',reviewed_at='',second_annotator='',adjudicator='',adjudication_note='')
            w.writerow(row)
            m+=1
    spec={'ner_label_schema_v2':list(ENTITY_SET),
          'status':'UNANNOTATED HUMAN REVIEW QUEUES ONLY',
          'existing_model_labels':'Nine silver entity kinds including DEADLINE; model needs future retraining for EDUCATION and broader DATE.',
          'ner_review_documents':n,'classification_review_documents':m,
          'gold_validated':False}
    (folder/'annotation_status_v2.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    print('Prepared pending review documents:',n,'pending classification rows:',m,'Gold: ZERO')

if __name__=='__main__':main()
