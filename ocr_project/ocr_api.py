import os
import openai
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# APIキーを設定
API_KEY = os.getenv("AVILEN_API_KEY")
API_URL = os.getenv("AVILEN_ENDPOINT")
if not API_KEY:
    raise ValueError(
        "API key must be set in the .env file or as an environment variable."
    )

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://llmapi.ops.avilen.co.jp/v1/winter_internship_2024/",
)


def ocr_pdf_api(pdf_path):
    try:
        # PDFファイルを読み込む
        with open(pdf_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        # プロンプトを指定
        prompt = "こんにちは\n"

        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        print(response)

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
