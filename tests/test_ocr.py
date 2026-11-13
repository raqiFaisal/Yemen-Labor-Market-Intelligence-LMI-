"""اختبار OCR على صور مُولَّدة (خلفية بيضاء، ملونة JPEG، داكنة بنص فاتح) + استخراج رابط الصورة من صفحة المعاينة.
    python -m tests.test_ocr      (يُتخطى إن لم يكن tesseract مثبتاً)
"""
import glob
import io
import shutil


def main():
    if not shutil.which("tesseract"):
        print("⚠ tesseract غير مثبت — تخطي اختبار OCR")
        raise SystemExit(0)
    from PIL import Image, ImageDraw, ImageFont
    from scripts.collectors.telegram_media_ocr import ocr_image, photo_urls
    fonts = [f for f in glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) if "DejaVuSans.ttf" in f]
    R = []

    def chk(c, label):
        R.append(bool(c))
        print(("✔ " if c else "✖ ") + label)

    html = ('<div class="tgme_widget_message" data-post="forus_taiz/123"><a class="tgme_widget_message_photo_wrap" '
            'style="width:800px;background-image:url(\'https://cdn4.telesco.pe/file/abc.jpg\')"></a></div>')
    chk(photo_urls(html, "forus_taiz") == {123: ["https://cdn4.telesco.pe/file/abc.jpg"]}, "استخراج رابط الصورة من المعاينة")
    if not fonts:
        print("⚠ لا خط عربي متاح لتوليد الصور — تخطي")
        raise SystemExit(0 if all(R) else 1)
    cases = [((1100, 420), "white", "black", ["مطلوب محاسب للعمل في شركة بعدن", "خبرة سنتين - بكالوريوس محاسبة"], "PNG"),
             ((1280, 900), (240, 200, 120), (20, 20, 60), ["إعلان وظيفة شاغرة", "تعلن مدارس النور عن حاجتها لمعلمة"], "JPEG"),
             ((900, 600), (15, 30, 80), (255, 255, 255), ["مطلوب مهندس مدني", "خبرة 5 سنوات - صنعاء"], "JPEG")]
    for size, bg, fg, lines, fmt in cases:
        img = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(img)
        for i, l in enumerate(lines):
            d.text((size[0] - 40, 80 + i * 130), l, fill=fg, font=ImageFont.truetype(fonts[0], 46), anchor="ra", direction="rtl")
        b = io.BytesIO()
        img.save(b, fmt)
        txt, conf = ocr_image(b.getvalue())
        chk(all(l in txt for l in lines) and conf > 60, f"OCR {fmt} خلفية {bg}: {conf}")
    print(f"\nالنتيجة: {sum(R)}/{len(R)} (صور مولّدة — الصور الحقيقية أصعب)")
    raise SystemExit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
