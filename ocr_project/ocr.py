import os
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
API_KEY = os.getenv("AVILEN_API_KEY")


def pdf_to_text(pdf_path):
    """PDFをOCR処理してテキストを抽出する"""
    images = convert_from_path(pdf_path)
    return "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images)
