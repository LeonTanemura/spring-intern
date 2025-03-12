import os
from ocr_project.ocr_api import ocr_pdf_api


def main():
    """OCRをAPI経由で実行する"""
    pdf_path = "data/sample1.pdf"
    if not os.path.exists(pdf_path):
        print(f"エラー: {pdf_path} が見つかりません")
        return

    print("=== APIへPDFを送信中... ===")
    api_response = ocr_pdf_api(pdf_path)

    if api_response:
        print("\n=== OCR結果 ===")
        print(api_response)


if __name__ == "__main__":
    main()
