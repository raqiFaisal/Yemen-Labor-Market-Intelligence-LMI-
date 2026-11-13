"""Generate reproducible no-Drive Colab training notebook for proper AraBERT fine-tuning."""
import json
from pathlib import Path


def md(s):
    return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}


def code(s):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(keepends=True)}


cells=[
md('''# LMI — Google Colab: تدريب AraBERT الحقيقي من البيانات الحالية\n
هذه الخلايا **لا تستخدم Google Drive**. الملف المضغوط يحتوي أيضاً على نماذج صغيرة PyTorch CPU **مدربة مسبقاً داخل هذه الحزمة على Silver** لتشغيل العرض محلياً؛ أما AraBERT فيحتاج تنزيل النموذج الأصلي من الإنترنت وتخصيصه فعلياً على GPU. **لا تُنسب إلى Gold البشري أي نتيجة**.
'''),
code('''from google.colab import files
from pathlib import Path
from zipfile import ZipFile
import os, sys, subprocess
uploaded=files.upload()
archives=[Path(x) for x in uploaded if x.lower().endswith('.zip')]
assert len(archives)==1, 'ارفع أرشيف المشروع النهائي ZIP'
zip_path=archives[0]
with ZipFile(zip_path) as z:
    for member in z.infolist():
        target=(Path('/content') / member.filename).resolve()
        if not target.is_relative_to(Path('/content')):
            raise ValueError('Unsafe file in ZIP: '+member.filename)
    z.extractall('/content')
root=Path('/content/LMI_PROJECT')
assert (root/'src/common.py').is_file(), 'ZIP must contain the top-level LMI_PROJECT folder'
os.chdir(root)
print('Loaded:',root)
'''),
code('''def run(*args):
    subprocess.run(list(map(str,args)),cwd=str(root),check=True)
run(sys.executable,'-m','pip','install','-r','requirements_runtime.txt','-r','requirements_dl.txt')
import torch
assert torch.cuda.is_available(), 'فعّل T4 GPU في Runtime > Change runtime type'
print('GPU:',torch.cuda.get_device_name(0))
run(sys.executable,'-m','scripts.verify_delivery')  # WARN is expected for missing human Gold
run(sys.executable,'-m','scripts.verify_model_artifacts')
'''),
md('''## تدريب النماذج الثلاثة (Fine-tuning)\n
ستُحفظ الأوزان داخل `models_dl/classification/*` و`models_dl/ner/`. شغّل هذه الخلية لتحويل نماذج CPU التجريبية إلى نماذج AraBERT حقيقية تستخدمها الواجهة **تلقائياً فقط عند اكتمال ملفات الأوزان والـ Tokenizer**. لا يتم التظاهر بأن النموذج درّب إذا فشل الأمر، و`check=True` يوقف الدفتر عند الخطأ.\n'''),
code('''for task in ('opportunity_type','geographic_scope'):
    modeldir=root/'models_dl'/'classification'/task
    present=(modeldir/'config.json').exists() and any((modeldir/x).exists() for x in
        ('model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json'))
    if not present:
        run(sys.executable,'-m','src.training.train_classifier','--task',task)
    else:
        print('Checkpoint exists, verifying by evaluation:',task)
    run(sys.executable,'-m','src.evaluation.evaluate_classifier','--task',task)
ner=root/'models_dl/ner'
present=(ner/'config.json').exists() and any((ner/x).exists() for x in
    ('model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json'))
if not present:
    run(sys.executable,'-m','src.training.train_ner')
run(sys.executable,'-m','src.evaluation.evaluate_ner')
print('Trained AraBERT checkpoints and SILVER-test evaluations saved.')
'''),
md('''## الاختبارات بعد التدريب\n
تُنفذ الاختبارات كاملة. لا تُعدّ أي مرحلة ناجحة إذا فشل اختبار. قد تكشف الاختبارات الإضافية أخطاء تنفيذ بيئية تحتاج إلى إصلاح، ولن يقوم الدفتر بإخفائها.\n'''),
code('''run(sys.executable,'-m','pytest','-q')
run(sys.executable,'-m','scripts.verify_delivery')
run(sys.executable,'-m','scripts.verify_model_artifacts')
'''),
md('''## تشغيل تطبيق Streamlit (اختياري)\n
التطبيق يعمل على منفذ 8501 ضمن جلسة Colab؛ من أجل عرض محلي مستقر أمام اللجنة يُفضَّل تشغيله على الحاسوب بعد تنزيل النسخة. لا يوجد هنا رابط خارجي وهمي.\n'''),
code('''import subprocess
app_process=subprocess.Popen([sys.executable,'-m','streamlit','run','app_dl/main.py',
    '--server.port=8501','--server.headless=true'],cwd=str(root))
print('Streamlit process PID:',app_process.pid,' port:8501')
'''),
md('''## حفظ المشروع مع أوزان AraBERT قبل إغلاق Colab\n
بعد التدريب تُحمَّل نسخة كاملة، ويمكن تشغيلها محلياً بـ `python -m streamlit run app_dl/main.py`. لا تدّعِ توافر Gold قبل مراجعة بشرية موثقة.\n'''),
code('''from shutil import make_archive
from google.colab import files
result=make_archive('/content/LMI_WITH_ARABERT','zip',root_dir='/content',base_dir='LMI_PROJECT')
print('Archive size:',Path(result).stat().st_size)
files.download(result)
''')]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                               'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
p=Path(__file__).resolve().parents[1]/'notebooks'/'LMI_Final_NoDrive.ipynb'
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
print(p,'cells:',len(cells))
