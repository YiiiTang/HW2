"""## Part B: 現代 AI 方法 (30分)

任務說明：使用 OpenAI API 完成相同的任務。**請勿把金鑰硬編碼在程式中**。
"""

import os
import json
from getpass import getpass
from openai import OpenAI, RateLimitError, APIError

# 建議使用環境變數或 getpass
api_key = os.environ.get("OPENAI_API_KEY") or getpass("請輸入您的 OpenAI API Key: ")

try:
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing OpenAI client: {e}")

"""### B-1: 語意相似度計算 (10分)"""

from openai import RateLimitError, APIError

def ai_similarity(text1, text2):
    """使用 OpenAI 模型判斷語意相似度
    要求：
    1) 設計適當 prompt
    2) 返回 0-100 的相似度分數（整數）
    3) 處理 API 錯誤
    """
    # TODO: 呼叫 OpenAI API，解析回傳結果並處理可能的錯誤
    # 提示: 使用 try-except 捕捉錯誤；回傳的結果需轉為 int

    system_prompt = """
    你是一個專注於評估語意相似度的 AI。
    你的任務是比較以下兩段文本。
    請提供一個 0 到 100 之間的分數 (0=完全不相關, 100=語意完全相同)。
    你必須只回應一個整數數字，不要包含任何解釋、文字或標點符號。
    """.strip()

    user_prompt = f"""
    文本1: "{text1}"
    文本2: "{text2}"
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )

        response_text = completion.choices[0].message.content.strip()

        try:
            score = int(response_text)
            return max(0, min(100, score))
        except ValueError:
            print(f"解析錯誤：模型回傳了非整數內容 '{response_text}'")
            return -1

    except RateLimitError:
        print("API 錯誤：已達到速率限制 (Rate Limit)。")
        return -1
    except APIError as e:
        print(f"API 錯誤：OpenAI 服務器回報錯誤: {e}")
        return -1
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return -1
    # raise NotImplementedError("請完成 ai_similarity() 的 API 呼叫與解析")

# 測試資料
text_a = "人工智慧是未來科技的趨勢"
text_b = "機器學習引領了AI的發展"
text_c = "今天天氣真好"

# 範例：完成後可取消註解
score1 = ai_similarity(text_a, text_b)
score2 = ai_similarity(text_a, text_c)
print(f'“{text_a}” 和 “{text_b}” 的相似度: {score1}')
print(f'“{text_a}” 和 “{text_c}” 的相似度: {score2}')

"""### B-2: AI 文本分類 (10分)"""

import json

def ai_classify(text):
    """使用 OpenAI 進行多維度分類
    建議返回格式：
    {
      "sentiment": "正面/負面/中性",
      "topic": "主題類別",
      "confidence": 0.95
    }
    """
    # TODO: 設計 prompt，呼叫 API，並解析回傳 JSON
    # 提示：在 prompt 明確要求模型回傳 JSON 字串，再用 json.loads() 解析

    system_prompt = """
    你是一個專業的文本分析 AI。
    你的任務是分析使用者提供的文本，並提供以下三項分類：
    1.  'sentiment' (情感): 必須是 '正面', '負面', 或 '中性' 其中之一。
    2.  'topic' (主題): 必須是 '科技', '運動', '美食', '旅遊', '娛樂', 或 '其他' 其中之一。
    3.  'confidence' (信心): 一個 0.0 到 1.0 之間的浮點數，代表你對此次分類的整體信心。

    你必須嚴格地只回傳一個有效的 JSON 物件，不要包含任何解釋性文字或 markdown 標記。
    """.strip()

    user_prompt = f"請分析以下文本：\n\"{text}\""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        response_text = completion.choices[0].message.content

        try:
            result_dict = json.loads(response_text)
            return result_dict

        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤: {e}. API 回應: {response_text}")
            return {"error": "JSON Decode Error", "response": response_text}

    except RateLimitError:
        print(f"API 錯誤：已達到速率限制 (Rate Limit)。 文本: {text[:20]}...")
        return {"error": "Rate Limit Error"}
    except APIError as e:
        print(f"API 錯誤：OpenAI 服務器回報錯誤: {e}")
        return {"error": "API Error", "message": str(e)}
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return {"error": "Unexpected Error", "message": str(e)}

    # raise NotImplementedError("請完成 ai_classify() 的 API 呼叫與解析")

# 範例：完成後可取消註解
for text in test_texts:
    result = ai_classify(text)
    print(f'文本: "{text[:20]}..." -> 分類結果: {result}')

"""### B-3: AI 自動摘要 (10分)"""

def ai_summarize(text, max_length=100):
    """使用 OpenAI 生成摘要
    要求：
    1) 可控制摘要長度
    2) 保留關鍵資訊
    3) 語句通順
    """
    # TODO: 設計 prompt，呼叫 API，並回傳摘要結果

    system_prompt = """
    你是一個專業的文本摘要 AI。
    你的任務是閱讀使用者提供的文章，並生成一段流暢、精準、保留所有關鍵資訊的摘要。
    摘要必須語意通順，並且符合使用者要求的長度限制。
    """.strip()

    user_prompt = f"""
    請將以下文章總結成一段**大約 {max_length} 字**（中文）的摘要：

    【文章】
    {text}
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=max_length * 2
        )

        summary = completion.choices[0].message.content.strip()

        return summary

    except RateLimitError:
        print(f"API 錯誤：已達到速率限制 (Rate Limit)。")
        return "[摘要生成失敗：速率限制]"
    except APIError as e:
        print(f"PI 錯誤：OpenAI 服務器回報錯誤: {e}")
        return f"[摘要生成失敗：API 錯誤 {e}]"
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return f"[摘要生成失敗： {e}]"

    # raise NotImplementedError("請完成 ai_summarize() 的 API 呼叫與解析")

# 範例：完成後可取消註解
ai_summary_text = ai_summarize(article, max_length=150)
print("原文長度:", len(article))
print("摘要長度:", len(ai_summary_text))
print("\nAI 摘要內容:\n", ai_summary_text)