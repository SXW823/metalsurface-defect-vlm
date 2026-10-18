import traceback
try:
    import pypdfium2 as pdfium
    from PIL import Image
    pdf = pdfium.PdfDocument(r"C:\MOV\metal_defect_vlm\output\Metal_Defect_VLM_Poster_Final.pdf")
    img = pdf[0].render(scale=2.2).to_pil()
    w, h = img.size
    img.save(r"C:\MOV\metal_defect_vlm\output\prev_full.png")
    img.crop((0, 0, w, int(h * 0.36))).save(r"C:\MOV\metal_defect_vlm\output\prev_top.png")
    img.crop((0, int(h * 0.34), w, int(h * 0.70))).save(r"C:\MOV\metal_defect_vlm\output\prev_mid.png")
    img.crop((0, int(h * 0.68), w, h)).save(r"C:\MOV\metal_defect_vlm\output\prev_bot.png")
    print("ok", w, h)
except Exception:
    traceback.print_exc()
