"""
Original file is located at
    https://colab.research.google.com/drive/1ajgo1vansebWWn4oulviw49ZLCeNcwoS
    
"""
import jieba
import numpy as np
import pandas as pd
from collections import Counter
import math

# 測試資料（可自行替換）
documents = [
    "人工智慧正在改變世界，機器學習是其核心技術",
    "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
    "今天天氣很好，適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康，每天都應該保持運動習慣"
]

# 中文斷詞
tokenized_documents = [list(jieba.cut(doc)) for doc in documents]
print("斷詞結果:")
for i, doc in enumerate(tokenized_documents, 1):
    print(f"Document {i}: {doc}")

"""#### 1. 手動實作 TF-IDF"""

def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)
    Args:
        word_dict: 詞彙計數字典 (e.g., {'人工智慧': 2, '世界': 1})
        total_words: 該文件的總詞數
    Returns:
        tf_dict: TF 值字典
    """
    # TODO: 實作 TF 計算
    # 提示：TF = (該詞在文件中出現的次數) / (文件總詞數)
    return {word: cnt / total_words for word, cnt in word_dict.items()}
    # raise NotImplementedError("請在此處完成 TF 計算")

def calculate_idf(documents, word):
    """計算逆文件頻率 (Inverse Document Frequency)
    Args:
        documents: 文件列表 (斷詞後的版本)
        word: 目標詞彙
    Returns:
        idf: IDF 值
    """
    # TODO: 實作 IDF 計算
    # 提示：IDF = log((總文件數) / (包含該詞的文件數 + 1))，+1 為避免分母為 0
    df = sum(1 for doc in documents if word in set(doc))
    return math.log(len(documents) / (df + 1))
    # raise NotImplementedError("請在此處完成 IDF 計算")

def calculate_tfidf(tokenized_documents):
    """計算 TF-IDF 主函數
    回傳：pandas.DataFrame，列為文件，欄為詞彙
    """
    # TODO:
    # 1) 遍歷所有文件，計算每個文件的 TF
    # 2) 建立詞彙庫 (vocabulary)
    # 3) 對詞彙庫中的每個詞，計算其 IDF
    # 4) 結合 TF 和 IDF 計算每個文件中每個詞的 TF-IDF 值
    # 5) 回傳 TF-IDF 矩陣 (pandas DataFrame)
    # raise NotImplementedError("請在此處完成 TF-IDF 主流程")
    vocabulary = set()
    for doc in tokenized_documents:
        vocabulary.update(doc)
    vocabulary = sorted(list(vocabulary))

    tf_list = []
    for doc in tokenized_documents:
        word_dict = {}
        for w in doc:
            word_dict[w] = word_dict.get(w, 0) + 1
        tf = calculate_tf(word_dict, len(doc))
        tf_list.append(tf)

    idf_dict = {word: calculate_idf(tokenized_documents, word)
                for word in vocabulary}

    tfidf_matrix = []
    for tf in tf_list:
        row = []
        for word in vocabulary:
            row.append(tf.get(word, 0) * idf_dict[word])
        tfidf_matrix.append(row)

    df = pd.DataFrame(tfidf_matrix, columns=vocabulary)
    return df

# 範例：完成後可取消註解
tfidf_matrix = calculate_tfidf(tokenized_documents)
print(tfidf_matrix.head())

"""#### 2. 使用 scikit-learn 實作"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TfidfVectorizer 需要以空格分隔的字串，所以我們先把斷詞結果接起來
processed_docs = [' '.join(doc) for doc in tokenized_documents]

# TODO: 使用 TfidfVectorizer 和 cosine_similarity 計算相似度矩陣
# 1) 初始化 TfidfVectorizer
# 2) fit_transform 文本資料
# 3) 使用 cosine_similarity 計算向量相似度

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_docs)
similarity_matrix = cosine_similarity(tfidf_matrix)

# raise NotImplementedError("請完成：scikit-learn 的 TF-IDF 與相似度計算")

"""#### 3. 視覺化（熱圖）"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

# TODO: 將上一步的 similarity_matrix 視覺化
similarity_matrix = np.array(similarity_matrix)

num_docs = len(processed_docs)
doc_labels = [f'文件 {i+1}' for i in range(num_docs)]

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap='viridis', xticklabels=range(1,6), yticklabels=range(1,6))
plt.title('文本相似度矩陣 (TF-IDF + Cosine Similarity)')
plt.xlabel('文件編號')
plt.ylabel('文件編號')
plt.savefig('results/tfidf_similarity_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

"""### A-2: 基於規則的文本分類 (15分)
任務說明：建立規則式分類器，不使用機器學習，純粹基於關鍵詞和規則。
1. 情感分類器 (8分)
2. 主題分類器 (7分)
"""

# 測試資料
test_texts = [
    "這家餐廳的牛肉麵真的太好吃了，湯頭濃郁，麵條Q彈，下次一定再來！",
    "最新的AI技術突破讓人驚豔，深度學習模型的表現越來越好",
    "這部電影劇情空洞，演技糟糕，完全是浪費時間",
    "每天慢跑5公里，配合適當的重訓，體能進步很多"
]

"""#### 1. 情感分類器"""

class RuleBasedSentimentClassifier:
    def __init__(self):
        # 建立正負面詞彙庫（可自行擴充）
        self.positive_words = ['好', '棒', '優秀', '喜歡', '推薦', '滿意', '開心', '值得', '精彩', '完美', '好吃', '濃郁', 'Q彈']
        self.negative_words = ['差', '糟', '失望', '討厭', '不推薦', '浪費', '無聊', '爛', '糟糕', '差勁', '空洞']
        self.negation_words = ['不', '沒', '無', '非', '別']
        self.adverb_degree_words = {'很': 2, '非常': 2.5, '超級': 3,'有點': 0.5, '稍微': 0.5}

    def classify(self, text):
        """
        分類邏輯（請自行實作）：
        1) 計算正負詞數量
        2) 處理否定詞（否定 + 正面 → 轉負；否定 + 負面 → 轉正）
        3) （可選）程度副詞加權
        回傳：'正面' / '負面' / '中性'
        """
        words = list(jieba.cut(text))

        score = 0
        last_word_is_negation = False
        last_word_is_adverb = 1.0

        for word in words:
            if word in self.positive_words:
                current_score = 1 * last_word_is_adverb
                if last_word_is_negation:
                    score -= current_score
                else:
                    score += current_score

                last_word_is_negation = False
                last_word_is_adverb = 1.0

            elif word in self.negative_words:
                current_score = 1 * last_word_is_adverb
                if last_word_is_negation:
                    score += current_score
                else:
                    score -= current_score

                last_word_is_negation = False
                last_word_is_adverb = 1.0

            elif word in self.negation_words:
                last_word_is_negation = True
                last_word_is_adverb = 1.0

            elif word in self.adverb_degree_words:
                last_word_is_adverb = self.adverb_degree_words[word]

            else:
                last_word_is_negation = False
                last_word_is_adverb = 1.0

        if score > 0:
            return '正面'
        elif score < 0:
            return '負面'
        else:
            return '中性'

        # TODO: 實作情感分類邏輯
        # raise NotImplementedError("請完成情感分類器 classify()")

# 範例：完成後可取消註解
sentiment_classifier = RuleBasedSentimentClassifier()
for text in test_texts:
    sentiment = sentiment_classifier.classify(text)
    print(f'文本: "{text[:20]}..." -> 情感: {sentiment}')

"""#### 2. 主題分類器"""

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI', '人工智慧', '電腦', '軟體', '程式', '演算法', '技術', '模型', '深度學習'],
            '運動': ['運動', '健身', '跑步', '游泳', '球類', '比賽', '慢跑', '體能'],
            '美食': ['吃', '食物', '餐廳', '美味', '料理', '烹飪', '牛肉麵', '湯頭'],
            '旅遊': ['旅行', '景點', '飯店', '機票', '觀光', '度假'],
            '娛樂': ['電影', '劇情', '演技', '音樂', '遊戲']
        }
        self.topic_keyword_sets = {topic: set(keywords) for topic, keywords in self.topic_keywords.items()}

    def classify(self, text):
        """返回最可能的主題（請實作關鍵詞計分）"""
        # TODO: 計算每個主題關鍵詞在文本中出現次數，回傳分數最高主題
        try:
            words = list(jieba.cut(text))
        except:
            words = []

        topic_scores = {topic: 0 for topic in self.topic_keywords}

        for word in words:
            for topic, keywords_set in self.topic_keyword_sets.items():
                if word in keywords_set:
                    topic_scores[topic] += 1

        best_topic = max(topic_scores, key=topic_scores.get)

        if topic_scores[best_topic] == 0:
            return '未知'
        else:
            return best_topic

        # raise NotImplementedError("請完成主題分類器 classify()")

# 範例：完成後可取消註解
topic_classifier = TopicClassifier()
for text in test_texts:
    topic = topic_classifier.classify(text)
    print(f'文本: "{text[:20]}..." -> 主題: {topic}')

"""### A-3: 統計式自動摘要 (15分)
任務說明：使用統計方法實作摘要系統，不依賴現代生成式 AI。
"""

from stopwordsiso import stopwords
import re
import jieba
from collections import Counter

# 測試文章（可自行替換）
article = (
    "人工智慧（AI）的發展正深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。\n"
    "在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。\n"
    "教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。\n"
    "然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。\n"
    "面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。\n"
)

class StatisticalSummarizer:
    def __init__(self):
        # 載入停用詞（繁體）
        self.stop_words = stopwords("zh")
        self.stop_words.update(['的', '在', '了', '是', '我', '你', '他', '她', '也', '都', '就', '個', '和', '與', '或', '為', '以', '等'])

    def _split_sentences(self, text):
        # 粗略中文分句：依 。！？ 及換行 分割
        sents = re.split(r"[。！？\n]+", text)
        return [s.strip() for s in sents if s.strip()]

    def _tokenize_and_filter(self, text):
        """輔助函數：斷詞並過濾停用詞與標點"""
        words = list(jieba.cut(text.lower()))
        filtered_words = []
        for word in words:
            word = word.strip()
            # 過濾停用詞、空字串、單純的標點或數字
            if word and word not in self.stop_words and not re.fullmatch(r"[\s\W\d_]+", word):
                filtered_words.append(word)
        return filtered_words

    def sentence_score(self, sentence, word_freq, idx, n_sent):
        """計算句子重要性分數（請自行設計）
        可考慮：高頻詞數量、句子位置(首尾加權)、句長懲罰、是否含數字／專有名詞等
        """
        # TODO: 實作句子評分邏輯
        sent_words = self._tokenize_and_filter(sentence)
        sent_len = len(sent_words)

        if sent_len < 4:
            return 0

        score = 0
        for word in sent_words:
            score += word_freq.get(word, 0)

        score = score / (sent_len ** 0.5)

        if idx == 0:
            score *= 1.5
        elif idx == n_sent - 1:
            score *= 1.2

        return score

        # raise NotImplementedError("請完成 sentence_score() 設計")

    def summarize(self, text, ratio=0.3):
        """
        生成摘要步驟建議：
        1) 分句
        2) 分詞並計算詞頻（過濾停用詞與標點）
        3) 計算每句分數
        4) 依 ratio 選取 Top-K 句子
        5) 依原文順序輸出摘要
        """
        # TODO: 實作摘要主流程

        sentences = self._split_sentences(text)
        n_sent = len(sentences)
        if n_sent == 0:
            return ""

        all_words = self._tokenize_and_filter(text)
        word_freq = Counter(all_words)

        sentence_scores = []
        for i, sent in enumerate(sentences):
            score = self.sentence_score(sent, word_freq, i, n_sent)
            sentence_scores.append((score, i, sent))

        k = max(1, int(n_sent * ratio)) # 至少選一句
        top_k_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:k]
        summary_sentences = sorted(top_k_sentences, key=lambda x: x[1])
        summary = "。".join([s[2] for s in summary_sentences]) + "。"

        return summary

        # raise NotImplementedError("請完成 summarize() 主流程")

# 範例：完成後可取消註解
summarizer = StatisticalSummarizer()
summary = summarizer.summarize(article, ratio=0.4)
print("原文長度:", len(article))
print("摘要內容:\n", summary)