import os
import cv2
import numpy as np
import pandas as pd
import openai
import base64
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# OpenAI APIキー設定
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

# PDFのパス
pdf_path = "data/sample2.pdf"

# **GPT-4o用のOCRプロンプト**
prompt = (
    "以下の画像は照明器具の仕様書（図面）です。\n"
    "この画像内から **「品番」「個数」「通し番号」** を抽出してください。  \n"
    "情報を整理し、次の表形式で出力してください。\n"
    "### **出力フォーマット**\n"
    "| 品番（商品番号） | 個数 | 通し番号 |\n"
    "|---------------|------|--------|\n"
    "| XXXX-YYYY-ZZZ | 2台  | 402    |\n"
    "| ABCD-1234-XYZ | 1台  | 150    |\n"
    "**注意点:**\n"
    "1. 画像内にある **品番らしき番号（例: NNFW42500K LE9）** を優先して抽出してください。\n"
    "2. **品番がない場合は、「品番なし」と記入してください。**\n"
    "3. 「個数」は **台数表記（例: 2台）** で記入してください。\n"
    "4. 通し番号（例: H402）があれば、そのまま記入してください。\n"
    "5. 画像に表形式の情報が含まれている場合、表ごと解析してください。\n"
)


# **PDFを画像に変換**
def pdf_to_images(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDFファイルが見つかりません: {pdf_path}")

    print(f"✅ {pdf_path} を画像に変換中...")
    images = convert_from_path(pdf_path)

    # 画像を保存（デバッグ用）
    os.makedirs("test_images", exist_ok=True)
    image_paths = []
    for i, img in enumerate(images):
        img_path = f"test_images/page_{i+1}.png"
        img.save(img_path, "PNG")
        image_paths.append(img_path)
        print(f"✅ {img_path} に画像を保存しました")

    return image_paths


# **画像を枠に沿って分割**
def split_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # エッジ検出（Canny法）
    edges = cv2.Canny(gray, 50, 150)

    # 輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 分割領域を取得
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:  # 小さすぎる領域は除外
            regions.append((x, y, w, h))

    # 上から順に並べる
    regions = sorted(regions, key=lambda r: (r[1], r[0]))

    # 分割した画像の保存
    os.makedirs("split_images", exist_ok=True)
    split_images = []

    for idx, (x, y, w, h) in enumerate(regions):
        sub_img = image[y : y + h, x : x + w]
        sub_img_path = f"split_images/part_{idx+1}.png"
        cv2.imwrite(sub_img_path, sub_img)
        split_images.append(sub_img_path)

    return split_images


# **画像をBase64エンコード**
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_str


# **GPT-4oを使ったOCR**
def ocr_with_openai(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    return response.choices[0].message.content


# **全処理の流れ**
image_paths = pdf_to_images(pdf_path)

all_results = []
for image_path in image_paths:
    split_images = split_image(image_path)
    for split_img in split_images:
        ocr_text = ocr_with_openai(split_img)
        all_results.append((split_img, ocr_text))

# **結果を表形式で表示**
df = pd.DataFrame(all_results, columns=["画像ファイル", "OCR結果"])
# **1. OCR結果をコンソールに表示**
print(df)

# **2. OCR結果をCSVに保存**
output_csv = "ocr_results.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"✅ OCR結果を {output_csv} に保存しました！")
