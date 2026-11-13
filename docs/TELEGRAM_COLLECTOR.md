# جامع Telegram — المرحلة 2

**الحالة:** الكود مكتمل ومختبر على HTML مصطنع (16/16). **لم يُشغَّل بعد على القنوات الحقيقية**؛ الأعداد الحقيقية تأتي من تشغيل Colab فقط.

## التشغيل
```
python -m scripts.collect_telegram --channels hr_yemen job_yemen --max-pages 3     # تجربة
python -m scripts.collect_telegram --channels hr_yemen job_yemen                  # تحديث + استكمال الأرشيف
python -m scripts.collect_telegram --all-enabled --mode update                    # الجديد فقط لكل القنوات المفعّلة
python -m scripts.collect_telegram --status
```

## ما يُحفظ (لكل قناة ملف مستقل)
| الملف | المحتوى |
|---|---|
| `data/raw/telegram/<ch>.csv` | منشور واحد لكل message_id: الرابط، `message_date` (تاريخ الرسالة)، `collected_at` / `last_seen_at`، `raw_text` كما هو، `text_hash`، `edited`، `forwarded_from`، وسائط، روابط، المشاهدات |
| `data/raw/telegram/<ch>_edits.csv` | النص السابق لأي منشور تغيّر نصه (لا يُفقد أي نص) |
| `data/raw/telegram/state/<ch>.json` | أقدم/أحدث رسالة واكتمال الأرشيف |
| `data/raw/logs/tg_<ch>_failures.csv` / `_runs.jsonl` | كل فشل وسببه / عدادات كل تشغيل |

- **تاريخ الرسالة ≠ تاريخ الإعلان:** `message_date` هو وقت النشر في القناة. التاريخ المكتوب داخل نص الإعلان (النشر/آخر موعد) يُستخرج في مرحلة التوحيد (5) إلى `published_date` / `application_deadline` منفصلاً.
- **المنشورات الخام ليست وظائف:** لا يقرؤها خط التدريب الحالي (يقرأ ملفات `data/raw/*.csv` في المستوى الأعلى فقط). التصنيف (JOB/TRAINING/...) والنطاق الجغرافي في المرحلة 5.

## القيود المعروفة
- المعاينة العامة لا تعرض تاريخ التعديل، فقط علامة «edited»؛ التعديل يُرصد بتغير `text_hash` عند إعادة الفحص (أحدث صفحتين في كل تحديث).
- الرسائل المحذوفة لا تظهر في المعاينة؛ تبقى نسختها المحفوظة سابقاً (لا حذف تلقائي).
- منشورات الصور بلا نص تُحفظ مع `media_only=True` (لا OCR في هذه المرحلة).
- Telethon (`--backend telethon`) يعطي بيانات أدق لكنه يحتاج حساب Telegram: `TG_API_ID`/`TG_API_HASH` من Colab Secrets أو متغيرات البيئة فقط، والجلسة في `.secrets/` المستثنى من Git.

## إضافة قناة جديدة
1. أضف إلى `data/sources.json` → `telegram`: `{"username": "اسم_القناة", "enabled": true, "notes": "..."}` (بدون @).
2. افحصها: `python -m scripts.probe_sources --only اسم_القناة` (تتأكد أن المعاينة العامة مفعّلة).
3. اجمع: `python -m scripts.collect_telegram --channels اسم_القناة --max-pages 3` ثم بدون حد.
