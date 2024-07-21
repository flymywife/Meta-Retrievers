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
import tiktoken
from rank_bm25 import BM25Okapi
import re

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
    prompt = f"Generate 6 questions about {entity} based on Who, What, When, Where, Why, and How. Ensure each question is a complete sentence."
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates 5W1H questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        temperature=0.7,
    )
    
    # Extract the content from the response
    content = response.choices[0].message.content.strip()
    
    # Split the content into individual queries
    queries = content.split('\n')
    
    # Remove the first three characters if they contain number, dot, and space
    cleaned_queries = [re.sub(r'^[\d. ]{3}', '', query.strip()) for query in queries]
    
    return {f"query_{i+1}": query for i, query in enumerate(cleaned_queries)}

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extend_answer(entity: str, query: str, current_answer: str, min_tokens: int, max_tokens: int) -> str:
    current_tokens = num_tokens_from_string(current_answer, "gpt-4o-mini-2024-07-18")
    
    if current_tokens >= min_tokens:
        return current_answer
    
    additional_tokens_needed = min_tokens - current_tokens
    prompt = f"The following is a partial answer to the question '{query}' about {entity}. Please extend this answer with additional relevant information. Add approximately {additional_tokens_needed} tokens:\n\n{current_answer}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extends answers with additional relevant information."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens - current_tokens,
        n=1,
        temperature=0.5,
    )
    
    extended_answer = current_answer + " " + response.choices[0].message.content.strip()
    
    return extend_answer(entity, query, extended_answer, min_tokens, max_tokens)

def generate_answer(entity: str, query: str, max_tokens: int) -> str:
    min_tokens = max(50, max_tokens - 100)
    
    prompt = f"Answer the following question about {entity} concisely: {query}"
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise answers to questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=1,
        temperature=0.5,
    )
    
    initial_answer = response.choices[0].message.content.strip()
    return extend_answer(entity, query, initial_answer, min_tokens, max_tokens)

def vectorize(text: str, model: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=[text],  # テキストをリストで囲む
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error in vectorize function: {str(e)}")
        logging.error(f"Input text: {text}")
        logging.error(f"Model: {model}")
        raise

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def process_entity(entity: str, embedding_model: str, max_tokens: int) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    queries = generate_5w1h_queries(entity)
    queries_data = {
        f"query_{i+1}": {"text": query, "vector": vectorize(query, embedding_model)}
        for i, (_, query) in enumerate(queries.items())
    }
    
    corpus_data = {
        f"answer_{i+1}": {"text": generate_answer(entity, query, max_tokens), "vector": vectorize(generate_answer(entity, query, max_tokens), embedding_model)}
        for i, (_, query) in enumerate(queries.items())
    }
    
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
    
    queries_data_no_vector = {key: {"text": data["text"]} for key, data in queries_data.items()}
    corpus_data_no_vector = {key: {"text": data["text"]} for key, data in corpus_data.items()}
    
    return queries_data, corpus_data, similarities_data, best_matches_data, queries_data_no_vector, corpus_data_no_vector


def preprocess_text(text: str) -> str:
    # 小文字化
    text = text.lower()
    # 句読点の除去（ただし、アポストロフィーは保持）
    text = re.sub(r'[^\w\s\']', '', text)
    # 余分な空白の除去
    text = ' '.join(text.split())
    return text

def match_queries_to_answers(queries, corpus):
    query_texts = [item['text'] for item in queries.values()]
    answer_texts = [item['text'] for item in corpus.values()]

    preprocessed_queries = [preprocess_text(text) for text in query_texts]
    preprocessed_answers = [preprocess_text(text) for text in answer_texts]

    bm25 = BM25Okapi(preprocessed_answers, k1=1.5, b=0.75)

    matches = []
    for query, orig_query in zip(preprocessed_queries, query_texts):
        scores = bm25.get_scores(query.split())
        best_match_index = scores.argmax()
        matches.append({
            "query": orig_query,
            "best_match_answer": answer_texts[best_match_index],
            "score": float(scores[best_match_index])
        })

    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches

def save_highest_keyword_score(entity: str, max_tokens: int, highest_score: float):
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"scores_{current_date}_{max_tokens}.json"
    file_path = os.path.join(HISTORY_FOLDER, file_name)
    
    data = {
        "entity": entity,
        "max_tokens": max_tokens,
        "highest_keyword_score": highest_score
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Highest keyword score saved to {file_path}")

def calculate_keyword_scores(queries: Dict, corpus: Dict) -> Dict[str, Dict[str, float]]:
    logging.debug(f"Queries: {queries}")
    logging.debug(f"Corpus: {corpus}")
    
    query_texts = [item['text'] for item in queries.values()]
    answer_texts = [item['text'] for item in corpus.values()]
    
    logging.debug(f"Query texts: {query_texts}")
    logging.debug(f"Answer texts: {answer_texts}")
    
    preprocessed_queries = [preprocess_text(text) for text in query_texts]
    preprocessed_answers = [preprocess_text(text) for text in answer_texts]
    
    logging.debug(f"Preprocessed queries: {preprocessed_queries}")
    logging.debug(f"Preprocessed answers: {preprocessed_answers}")
    
    # BM25Okapiオブジェクトを作成する前に、答えのテキストをトークン化します
    tokenized_answers = [answer.split() for answer in preprocessed_answers]
    
    bm25 = BM25Okapi(tokenized_answers)

    scores = {}
    for query_key, query in zip(queries.keys(), preprocessed_queries):
        query_tokens = query.split()
        query_scores = bm25.get_scores(query_tokens)
        logging.debug(f"Raw scores for {query_key}: {query_scores}")
        scores[query_key] = {f"answer_{i+1}": float(score) for i, score in enumerate(query_scores)}
    
    return scores

def save_keyword_scores(entity: str, max_tokens: int, keyword_scores: Dict[str, Dict[str, float]]):
    current_date = datetime.now().strftime("%Y%m%d")
    file_name = f"scores_{current_date}_{max_tokens}.json"
    file_path = os.path.join(HISTORY_FOLDER, file_name)
    
    data = {
        "entity": entity,
        "max_tokens": max_tokens,
        "keyword_scores": keyword_scores
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Keyword scores saved to {file_path}")


def integrated_interface(entity: str, embedding_model: str, max_tokens: int) -> Tuple[str, str, str, str, str]:
    try:
        logging.info(f"Processing entity: {entity} with model: {embedding_model} and max tokens: {max_tokens}")
        
        queries_data, corpus_data, similarities_data, best_matches_data, queries_data_no_vector, corpus_data_no_vector = process_entity(entity, embedding_model, max_tokens)
           
        save_json_file({"entity": entity, "queries": queries_data}, "queries", embedding_model, max_tokens)
        save_json_file({"entity": entity, "corpus": corpus_data}, "corpus", embedding_model, max_tokens)
        save_json_file({"entity": entity, "similarities": similarities_data}, "similarities", embedding_model, max_tokens)
        save_json_file({"entity": entity, "best_matches": best_matches_data}, "best_matches", embedding_model, max_tokens)
        
        save_json_file({"entity": entity, "queries": queries_data_no_vector}, f"queries_for_{entity}", embedding_model, max_tokens)
        save_json_file({"entity": entity, "corpus": corpus_data_no_vector}, f"corpus_for_{entity}", embedding_model, max_tokens)
        
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
        
        # キーワードスコアの計算と保存
        keyword_scores = calculate_keyword_scores(queries_data_no_vector, corpus_data_no_vector)
        save_json_file({"entity": entity, "max_tokens": max_tokens, "keyword_scores": keyword_scores}, "scores", embedding_model, max_tokens)
        
        keyword_matches_summary = "\nKeyword-based Matches:\n"
        for query_key, scores in keyword_scores.items():
            keyword_matches_summary += f"{query_key}:\n"
            for answer_key, score in scores.items():
                keyword_matches_summary += f"  {answer_key}: {score:.4f}\n"
            keyword_matches_summary += "\n"
        
        logging.info("Entity processing completed successfully")
        return queries_and_answers_text, similarity_summary, json.dumps({"entity": entity, "similarities": similarities_data}, indent=2), best_matches_summary, keyword_matches_summary
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return error_message, "", "{}", "", ""


iface = gr.Interface(
    fn=integrated_interface,
    inputs=[
        gr.Textbox(label="Enter an entity (person, place, event, etc.)"),
        gr.Dropdown(choices=EMBEDDING_MODELS, label="Select Embedding Model", value=EMBEDDING_MODELS[0]),
        gr.Slider(minimum=50, maximum=500, step=10, label="Max Tokens for Answer", value=150)
    ],
    outputs=[
        gr.Textbox(label="Generated Queries and Answers"),
        gr.Textbox(label="Cosine Similarities Summary"),
        gr.JSON(label="Detailed Similarities"),
        gr.Textbox(label="Best Matches Summary (Embedding-based)"),
        gr.Textbox(label="Keyword-based Matches Summary")
    ],
    title="Integrated 5W1H RAG Model and Keyword Matching for Entities",
    description="Enter an entity, select an embedding model, and set the maximum tokens for answers. This will generate 5W1H queries, answers, calculate similarities, find best matches using embeddings, and perform keyword-based matching."
)

if __name__ == "__main__":
    iface.launch(debug=True)