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

# Load .env file
load_dotenv(find_dotenv())

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# Create results folder
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Available embedding models
EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]

def save_json_file(data: Dict, file_prefix: str, folder: str, max_tokens: int, temperature: float):
    """Save data as a JSON file in the specified folder with date, max_tokens, and temperature in the filename"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_prefix}_{current_time}_{str(max_tokens)}_{temperature:.1f}.json"
    file_path = os.path.join(folder, file_name)
    
    os.makedirs(folder, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Data saved to {file_path}")

def generate_5w1h_queries(entity: str, max_tokens: int, temperature: float) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    prompt = f"Generate 6 questions about {entity} based on Who, What, When, Where, Why, and How. Ensure each question is a complete sentence."
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Who, What, When, Where, Why, and How questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        n=1,
        temperature=temperature,
        seed=42,
    )
    
    content = response.choices[0].message.content.strip()
    queries = content.split('\n')
    cleaned_queries = [re.sub(r'^[\d. ]{3}', '', query.strip()) for query in queries]
    
    return {f"query_{i+1}": query for i, query in enumerate(cleaned_queries)}

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extend_answer(entity: str, query: str, current_answer: str, min_tokens: int, max_tokens: int, temperature: float) -> str:
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
        temperature=temperature,
        seed=42,
    )
    
    extended_answer = current_answer + " " + response.choices[0].message.content.strip()
    
    return extend_answer(entity, query, extended_answer, min_tokens, max_tokens, temperature)

def generate_answer(entity: str, query: str, max_tokens: int, temperature: float) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
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
        temperature=temperature,
        seed=42,
    )
    
    initial_answer = response.choices[0].message.content.strip()
    return extend_answer(entity, query, initial_answer, min_tokens, max_tokens, temperature)

def vectorize(text: str, model: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=[text],
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

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    text = ' '.join(text.split())
    return text

def calculate_bm25_scores(queries: Dict, corpus: Dict) -> Tuple[Dict[str, Dict[str, float]], Dict]:
    query_texts = [item['text'] for item in queries.values()]
    answer_texts = [item['text'] for item in corpus.values()]
    
    preprocessed_queries = [preprocess_text(text) for text in query_texts]
    preprocessed_answers = [preprocess_text(text) for text in answer_texts]
    
    tokenized_answers = [answer.split() for answer in preprocessed_answers]
    bm25 = BM25Okapi(tokenized_answers)

    scores = {}
    best_matches = {}
    for query_key, query in zip(queries.keys(), preprocessed_queries):
        query_tokens = query.split()
        query_scores = bm25.get_scores(query_tokens)
        scores[query_key] = {f"answer_{i+1}": float(score) for i, score in enumerate(query_scores)}
        
        best_answer_key = max(scores[query_key], key=scores[query_key].get)
        best_score = scores[query_key][best_answer_key]
        best_matches[query_key] = {
            "query_text": queries[query_key]["text"],
            "best_answer_key": best_answer_key,
            "best_answer_text": corpus[best_answer_key]["text"],
            "bm25_score": best_score
        }
    
    return scores, best_matches

def find_best_matches(queries_data, corpus_data, embedding_results):
    best_matches = {}
    for model, similarities in embedding_results.items():
        best_matches[model] = {}
        for query_key, query_similarities in similarities.items():
            best_answer_key = max(query_similarities, key=query_similarities.get)
            best_similarity = query_similarities[best_answer_key]
            best_matches[model][query_key] = {
                "query_text": queries_data[query_key]["text"],
                "best_answer_key": best_answer_key,
                "best_answer_text": corpus_data[best_answer_key]["text"],
                "cosine_similarity": best_similarity
            }
    return best_matches

def process_entity(entity: str, max_tokens: int, temperature: float) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    RESULTS_FOLDER = f"results_{entity.replace(' ', '_')}"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    queries = generate_5w1h_queries(entity, max_tokens,temperature)    # Create result folder name that includes entity name
    queries_data = {
        f"query_{i+1}": {"text": query}
        for i, (_, query) in enumerate(queries.items())
    }
    
    corpus_data = {
        f"answer_{i+1}": {"text": generate_answer(entity, query, max_tokens,temperature)}
        for i, (_, query) in enumerate(queries.items())
    }
    
    # Save queries and corpus
    save_json_file({"entity": entity, "queries": queries_data}, "queries", RESULTS_FOLDER, max_tokens, temperature)
    save_json_file({"entity": entity, "corpus": corpus_data}, "corpus", RESULTS_FOLDER, max_tokens, temperature)
    
    
    # Process for each embedding model
    embedding_results = {}
    for model in EMBEDDING_MODELS:
        model_folder = os.path.join(RESULTS_FOLDER, model)
        os.makedirs(model_folder, exist_ok=True)

        corpus_vectors = {
            key: {
                "text": data["text"],
                "vector": vectorize(data["text"], model)
            }
            for key, data in corpus_data.items()
        }
        queries_vectors = {
            key: {
                "text": data["text"],
                "vector": vectorize(data["text"], model)
            }
            for key, data in queries_data.items()
        }
        
        cosine_similarities = {}
        for query_key, query_data in queries_vectors.items():
            cosine_similarities[query_key] = {}
            for answer_key, answer_data in corpus_vectors.items():
                similarity = cosine_similarity(query_data["vector"], answer_data["vector"])
                cosine_similarities[query_key][answer_key] = similarity
        
        embedding_results[model] = cosine_similarities
        
        # Save cosine similarities
        save_json_file({"entity": entity, "cosine_similarities": cosine_similarities}, 
                   "cosine_similarities", model_folder, max_tokens, temperature)
        
        # Save vector data with associated text
        save_json_file({"entity": entity, "corpus_vectors": corpus_vectors, "query_vectors": queries_vectors},
                   "vectors", model_folder, max_tokens, temperature)

    
    # Process for BM25
    bm25_folder = os.path.join(RESULTS_FOLDER, "BM25")
    os.makedirs(bm25_folder, exist_ok=True)
    bm25_scores, bm25_best_matches = calculate_bm25_scores(queries_data, corpus_data)
    save_json_file({"entity": entity, "scores": bm25_scores}, "scores", bm25_folder, max_tokens, temperature)
    save_json_file({"entity": entity, "best_matches": bm25_best_matches}, "best_matches", bm25_folder, max_tokens, temperature)

    
    # Find and save best matches for each model
    best_matches = find_best_matches(queries_data, corpus_data, embedding_results)
    for model, model_best_matches in best_matches.items():
        model_folder = os.path.join(RESULTS_FOLDER, model)
        save_json_file({"entity": entity, "best_matches": model_best_matches}, 
                       "best_matches", model_folder, max_tokens, temperature)
    
    return queries_data, corpus_data, embedding_results, bm25_scores, best_matches, bm25_best_matches


def batch_process(entity: str, max_tokens: int, temperature: float, step: int = 100) -> List[Dict]:
    results = []
    for tokens in range(step, max_tokens + 1, step):
        logging.info(f"Processing entity: {entity} with max tokens: {tokens}")
        queries_data, corpus_data, embedding_results, bm25_scores, best_matches, bm25_best_matches = process_entity(entity, tokens, temperature)
        results.append({
            "max_tokens": tokens,
            "queries_data": queries_data,
            "corpus_data": corpus_data,
            "embedding_results": embedding_results,
            "bm25_scores": bm25_scores,
            "best_matches": best_matches,
            "bm25_best_matches": bm25_best_matches
        })
    return results

def integrated_interface(entity: str, max_tokens: int, temperature: float) -> Tuple[str, str, str, str, str]:
    try:
        step = 100
        max_tokens = (max_tokens // step) * step 
        batch_results = batch_process(entity, max_tokens, temperature, step)
        
        queries_and_answers_text = ""
        embedding_summary = ""
        bm25_summary = ""
        best_matches_summary = ""
        
        for result in batch_results:
            current_max_tokens = result["max_tokens"]
            queries_and_answers_text += f"\n--- Results for max tokens: {current_max_tokens} ---\n\n"
            queries_and_answers_text += f"Generated 5W1H Queries and Answers for '{entity}' (max tokens: {current_max_tokens}):\n\n"
            for query_key, query_data in result["queries_data"].items():
                answer_key = f"answer_{query_key.split('_')[1]}"
                queries_and_answers_text += f"{query_key}: {query_data['text']}\n"
                queries_and_answers_text += f"{answer_key}: {result['corpus_data'][answer_key]['text']}\n\n"
            
            embedding_summary += f"\n--- Embedding-based Similarities for max tokens: {current_max_tokens} ---\n"
            for model, similarities in result["embedding_results"].items():
                embedding_summary += f"\n{model}:\n"
                for query_key, query_similarities in similarities.items():
                    embedding_summary += f"{query_key}:\n"
                    for answer_key, similarity in query_similarities.items():
                        embedding_summary += f"  {answer_key}: {similarity:.4f}\n"
                    embedding_summary += "\n"
            
            bm25_summary += f"\n--- BM25 Scores for max tokens: {current_max_tokens} ---\n"
            for query_key, scores in result["bm25_scores"].items():
                bm25_summary += f"{query_key}:\n"
                for answer_key, score in scores.items():
                    bm25_summary += f"  {answer_key}: {score:.4f}\n"
                bm25_summary += "\n"
            
            best_matches_summary += f"\n--- Best Matches for max tokens: {current_max_tokens} ---\n"
            for model, model_best_matches in result["best_matches"].items():
                best_matches_summary += f"\n{model}:\n"
                for query_key, match_data in model_best_matches.items():
                    best_matches_summary += f"{query_key}:\n"
                    best_matches_summary += f"  Query: {match_data['query_text']}\n"
                    best_matches_summary += f"  Best Answer ({match_data['best_answer_key']}): {match_data['best_answer_text']}\n"
                    best_matches_summary += f"  Cosine Similarity: {match_data['cosine_similarity']:.4f}\n\n"
            
            best_matches_summary += f"\nBM25 Best Matches for max tokens: {current_max_tokens}:\n"
            for query_key, match_data in result["bm25_best_matches"].items():
                best_matches_summary += f"{query_key}:\n"
                best_matches_summary += f"  Query: {match_data['query_text']}\n"
                best_matches_summary += f"  Best Answer ({match_data['best_answer_key']}): {match_data['best_answer_text']}\n"
                best_matches_summary += f"  BM25 Score: {match_data['bm25_score']:.4f}\n\n"

        logging.info("Batch processing completed successfully")
        return queries_and_answers_text, embedding_summary, bm25_summary, best_matches_summary, json.dumps({"entity": entity, "batch_results": batch_results}, indent=2)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return error_message, "", "", "", "{}"

iface = gr.Interface(
    fn=integrated_interface,
    inputs=[
        gr.Textbox(label="Enter an entity (person, place, event, etc.)"),
        gr.Slider(minimum=100, maximum=2000, step=100, label="Max Tokens for Answer", value=100),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.5)
    ],
    outputs=[
        gr.Textbox(label="Generated Queries and Answers"),
        gr.Textbox(label="Embedding-based Similarities Summary"),
        gr.Textbox(label="BM25 Scores Summary"),
        gr.Textbox(label="Best Matches Summary"),
        gr.JSON(label="Detailed Results")
    ],
    title="Six-Dimensional Query Tool with Batch Processing",
    description="Generate Six-Dimensional Queries (Who, What, When, Where, Why, How) and their corresponding corpus for any given entity. Processes in 100-token increments up to the specified max tokens. Adjust temperature to control randomness."
)

if __name__ == "__main__":
    iface.launch(debug=True)