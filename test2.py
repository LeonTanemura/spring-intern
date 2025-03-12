import os
import openai
import base64
import json
import time
import pandas as pd
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# APIキーとエンドポイントを設定
API_KEY = os.getenv("AVILEN_API_KEY")
API_URL = os.getenv("AVILEN_ENDPOINT")
SAMPLE_NUMBER = "3"

if not API_KEY:
    raise ValueError(
        "API key must be set in the .env file or as an environment variable."
    )

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://llmapi.ops.avilen.co.jp/v1/winter_internship_2024/",
)

# OCR対象の画像フォルダ
image_dir = f"split_images/{SAMPLE_NUMBER}/"


# 画像をBase64に変換する関数
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_str


# すべての画像を処理
ocr_results = []
index = 0
total_start_time = time.time()  # 総処理時間の開始

while True:
    filename = f"image{index}.png"  # image0.png, image1.png, image2.png ...
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        print(f"✅ すべての画像を処理しました。（最後のindex: {index-1}）")
        break  # 画像が存在しなくなったらループ終了

    # 画像をBase64エンコード
    base64_image = encode_image(image_path)

    # OCRプロンプト
    prompt = (
        "以下の画像は照明器具の仕様書（図面）です。\n"
        "画像を解析し、JSON形式で出力してください。照明に関する情報を抽出し、"
        "次のキーを持つJSONオブジェクトとして出力してください。 \n"
        "情報を整理し、次の特徴を持ったJSON形式で出力してください。\n"
        "### 出力フォーマット\n"
        "{\n"
        f"   'filename': '{filename}',\n"
        "   'hinban': '<商品番号または公共施設型番>',\n"
        "   'num_items': '<商品の個数>',\n"
        "   'serial_num': '<通し番号>',\n"
        "   'other': '<その他の関連情報>'\n"
        "}\n\n"
        "### 具体的な指示：\n"
        "filename: 入力画像名\n"
        "hinban: 照明の商品番号または公共施設型番を抽出してください。\n"
        "num_items: 照明の必要個数を数えてください。\n"
        "serial_num: 左上に通し番号が明記されていれば抽出してください。\n"
        "other: その他の照明に関連する情報（ブランド名、仕様、設置場所など）があれば含めてください。\n"
        "### 出力例：\n"
        f"   'filename': '{filename}',\n"
        "   'hinban': 'X323FS',\n"
        "   'num_items': '2',\n"
        "   'serial_num': 'A301',\n"
        "   'other': 'LED照明、色温度5000K'\n"
        "}\n\n"
        "**注意点:**\n"
        "1. 画像内にある **商品番号らしき番号（例: NNFW42500K LE9）** を優先して抽出してください。\n"
        "2. 商品番号がない場合は、**「商品番号なし」**と記入してください。\n"
        "3. 商品番号が複数ある場合は、すべて読み取るようにしてください。\n"
        "4. 通し番号**（例: H402）**があれば、そのまま記入してください。\n"
        "5. 商品番号に英数字以外の文字が含まれている場合は、英数字以外の文字を除いて、**'other': その他の情報**の部分に追加するようにしてください。\n"
    )

    # OCR処理の開始時間を記録
    start_time = time.time()

    # GPT-4o に画像を送信
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

    # OCR処理の終了時間を記録
    end_time = time.time()
    elapsed_time = end_time - start_time  # OCR処理時間を計算

    # OCR結果を取得
    ocr_text = response.choices[0].message.content
    print(f"filename: {filename}, time: {elapsed_time}")
    print(ocr_text)

    try:
        # 生成AIの出力をJSONとして解析
        parsed_json = json.loads(ocr_text)
        parsed_json["filename"] = filename  # 画像ファイル名を追加
        parsed_json["processing_time"] = round(elapsed_time, 2)  # 処理時間を追加
        ocr_results.append(parsed_json)
    except json.JSONDecodeError:
        # print(f"⚠️ JSONパースエラー: {filename}")
        ocr_results.append(
            {
                "filename": filename,
                "error": "JSONパース失敗",
                "raw_output": ocr_text,
                "processing_time": round(elapsed_time, 2),  # 失敗時も処理時間を記録
            }
        )

    # 次のファイルへ
    index += 1

# OCR結果をJSONに保存
output_json = f"result/ocr_results_{SAMPLE_NUMBER}.json"

# JSONファイルに保存
with open(output_json, "w", encoding="utf-8") as json_file:
    json.dump(ocr_results, json_file, ensure_ascii=False, indent=4)

# 総処理時間を記録
total_end_time = time.time()
total_time = round(total_end_time - total_start_time, 2)

print(f"✅ OCR結果を {output_json} に保存しました！")
print(f"📊 総処理時間: {total_time} 秒")

# OCR結果を表示
# print(json.dumps(ocr_results, ensure_ascii=False, indent=4))
