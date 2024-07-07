import gradio as gr
from openai import OpenAI
from typing import List, Tuple, Dict
import os
import logging
import json
from datetime import datetime
import numpy as np
from dotenv import load_dotenv, find_dotenv
import openai
import traceback

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# .env ファイルを探して読み込む
load_dotenv(find_dotenv())

# OpenAI クライアントの初期化
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# historyフォルダの作成
HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# 利用可能なembeddingモデル
EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small"]

def save_json_file(data: Dict, file_prefix: str, model_name: str, max_tokens: int):
    """データをタイムスタンプ付きのJSONファイルとしてhistory/model名フォルダに保存する"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(HISTORY_FOLDER, model_name)
    os.makedirs(model_folder, exist_ok=True)
    file_name = f"{file_prefix}_{current_time}_{str(max_tokens)}.json"
    file_path = os.path.join(model_folder, file_name)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Data saved to {file_path}")
    
def generate_5w1h_queries(entity: str) -> Dict[str, str]:
    """固有名詞に基づいて5W1Hのクエリを生成する"""
    prompt = f"Generate 6 questions about {entity} based on Who, What, When, Where, Why, and How. Ensure each question is a complete sentence."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates 5W1H questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        temperature=0.7,
    )
    queries = response.choices[0].message.content.strip().split("\n")
    return {f"query_{i+1}": query.strip() for i, query in enumerate(queries)}

def generate_answer(entity: str, query: str, max_tokens: int) -> str:
    """クエリに対する回答を生成する"""
    prompt = f"Answer the following question about {entity} concisely: {query}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise answers to questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def vectorize(text: str, model: str) -> List[float]:
    """テキストを指定されたモデルでベクトル化する"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """2つのベクトル間のコサイン類似度を計算する"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def process_entity(entity: str, embedding_model: str, max_tokens: int) -> Tuple[Dict, Dict, Dict, Dict]:
    """固有名詞を処理し、クエリ、コーパス、類似度、ベストマッチを生成する"""
    # クエリの生成とベクトル化
    queries = generate_5w1h_queries(entity)
    queries_data = {
        key: {"text": query, "vector": vectorize(query, embedding_model)}
        for key, query in queries.items()
    }
    
    # 各クエリに対する回答の生成とベクトル化
    corpus_data = {
        key: {"text": generate_answer(entity, query, max_tokens), "vector": vectorize(generate_answer(entity, query, max_tokens), embedding_model)}
        for key, query in queries.items()
    }
    
    # 類似度の計算（クエリとアンサーの間）
    similarities_data = {}
    best_matches_data = {}
    for query_key, query_data in queries_data.items():
        similarities_data[query_key] = {}
        max_similarity = -1
        best_match_key = ""
        for answer_key, answer_data in corpus_data.items():
            similarity = cosine_similarity(query_data["vector"], answer_data["vector"])
            similarities_data[query_key][answer_key] = similarity
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_key = answer_key
        
        best_matches_data[query_key] = {
            "query_text": query_data["text"],
            "best_answer_text": corpus_data[best_match_key]["text"],
            "similarity": max_similarity
        }
    
    return queries_data, corpus_data, similarities_data, best_matches_data

def gradio_interface(entity: str, embedding_model: str, max_tokens: int) -> Tuple[str, str, str, str]:
    """Gradioインターフェース用の関数"""
    try:
        logging.info(f"Processing entity: {entity} with model: {embedding_model} and max tokens: {max_tokens}")
        
        queries_data, corpus_data, similarities_data, best_matches_data = process_entity(entity, embedding_model, max_tokens)
        
        # Save data to separate JSON files with timestamps in the history/model_name folder
        save_json_file({"entity": entity, "queries": queries_data}, "queries", embedding_model, max_tokens)
        save_json_file({"entity": entity, "corpus": corpus_data}, "corpus", embedding_model, max_tokens)
        save_json_file({"entity": entity, "similarities": similarities_data}, "similarities", embedding_model, max_tokens)
        save_json_file({"entity": entity, "best_matches": best_matches_data}, "best_matches", embedding_model, max_tokens)
        
        # Prepare output for Gradio interface
        queries_and_answers_text = f"Generated 5W1H Queries and Answers for '{entity}' using {embedding_model} (max tokens: {max_tokens}):\n\n"
        for (query_key, query_data), (answer_key, answer_data) in zip(queries_data.items(), corpus_data.items()):
            queries_and_answers_text += f"{query_key}: {query_data['text']}\n"
            queries_and_answers_text += f"{answer_key}: {answer_data['text']}\n\n"
        
        similarity_summary = "\nCosine Similarities between Queries and Answers:\n"
        for query_key, similarities in similarities_data.items():
            similarity_summary += f"{query_key}:\n"
            for answer_key, similarity in similarities.items():
                similarity_summary += f"  {answer_key}: {similarity:.4f}\n"
            similarity_summary += "\n"
        
        best_matches_summary = "\nBest Matches (Query-Answer Pairs with Highest Similarity):\n"
        for query_key, match_data in best_matches_data.items():
            best_matches_summary += f"{query_key}:\n"
            best_matches_summary += f"  Query: {match_data['query_text']}\n"
            best_matches_summary += f"  Best Answer: {match_data['best_answer_text']}\n"
            best_matches_summary += f"  Similarity: {match_data['similarity']:.4f}\n\n"
        
        logging.info("Entity processing completed successfully")
        return queries_and_answers_text, similarity_summary, json.dumps(similarities_data, indent=2), best_matches_summary
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return error_message, "", "{}", ""

# Gradioインターフェースの作成
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter an entity (person, place, event, etc.)"),
        gr.Dropdown(choices=EMBEDDING_MODELS, label="Select Embedding Model", value=EMBEDDING_MODELS[0]),
        gr.Slider(minimum=50, maximum=500, step=10, label="Max Tokens for Answer", value=150)
    ],
    outputs=[
        gr.Textbox(label="Generated Queries and Answers"),
        gr.Textbox(label="Cosine Similarities Summary"),
        gr.JSON(label="Detailed Similarities"),
        gr.Textbox(label="Best Matches Summary")
    ],
    title="5W1H RAG Model for Entities",
    description="Enter an entity, select an embedding model, and set the maximum tokens for answers to generate 5W1H queries, answers, calculate similarities, and find best matches."
)

# アプリケーションの起動（デバッグモードを有効に）
if __name__ == "__main__":
    iface.launch(debug=True)