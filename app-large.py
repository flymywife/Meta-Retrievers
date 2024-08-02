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

# Available embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

def save_json_file(data: Dict, file_prefix: str, folder: str, max_tokens: int):
    """Save data as a JSON file in the specified folder with date and max_tokens in the filename"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{file_prefix}_{current_time}_{str(max_tokens)}.json"
    file_path = os.path.join(folder, file_name)
    
    os.makedirs(folder, exist_ok=True)
    
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
        temperature=0.5,
    )
    
    content = response.choices[0].message.content.strip()
    queries = content.split('\n')
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

def vectorize(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error in vectorize function: {str(e)}")
        logging.error(f"Input text: {text}")
        logging.error(f"Model: {EMBEDDING_MODEL}")
        raise

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def process_entity(entity: str, max_tokens: int) -> Tuple[Dict, Dict, Dict, Dict]:
    # エンティティ名を含む結果フォルダ名を作成
    RESULTS_FOLDER = f"results_{entity.replace(' ', '_')}"
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    queries = generate_5w1h_queries(entity)
    queries_data = {
        f"query_{i+1}": {"text": query}
        for i, (_, query) in enumerate(queries.items())
    }
    
    corpus_data = {
        f"answer_{i+1}": {"text": generate_answer(entity, query, max_tokens)}
        for i, (_, query) in enumerate(queries.items())
    }
    
    # Save queries and corpus
    save_json_file({"entity": entity, "queries": queries_data}, "queries", RESULTS_FOLDER, max_tokens)
    save_json_file({"entity": entity, "corpus": corpus_data}, "corpus", RESULTS_FOLDER, max_tokens)
    
    # Process for embedding model
    model_folder = os.path.join(RESULTS_FOLDER, EMBEDDING_MODEL)
    os.makedirs(model_folder, exist_ok=True)

    corpus_vectors = {
        key: {
            "text": data["text"],
            "vector": vectorize(data["text"])
        }
        for key, data in corpus_data.items()
    }
    queries_vectors = {
        key: {
            "text": data["text"],
            "vector": vectorize(data["text"])
        }
        for key, data in queries_data.items()
    }
    
    cosine_similarities = {}
    for query_key, query_data in queries_vectors.items():
        cosine_similarities[query_key] = {}
        for answer_key, answer_data in corpus_vectors.items():
            similarity = cosine_similarity(query_data["vector"], answer_data["vector"])
            cosine_similarities[query_key][answer_key] = similarity
    
    # Save cosine similarities
    save_json_file({"entity": entity, "cosine_similarities": cosine_similarities}, 
                   "cosine_similarities", model_folder, max_tokens)
    
    # Save vector data with associated text
    save_json_file({"entity": entity, "corpus_vectors": corpus_vectors, "query_vectors": queries_vectors},
                   "vectors", model_folder, max_tokens)
    
    # Find and save best matches
    best_matches = find_best_matches(queries_data, corpus_data, {EMBEDDING_MODEL: cosine_similarities})
    save_json_file({"entity": entity, "best_matches": best_matches[EMBEDDING_MODEL]}, 
                   "best_matches", model_folder, max_tokens)
    
    return queries_data, corpus_data, cosine_similarities, best_matches

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

def integrated_interface(entity: str, max_tokens: int) -> Tuple[str, str, str, str]:
    try:
        logging.info(f"Processing entity: {entity} with max tokens: {max_tokens}")
        
        queries_data, corpus_data, cosine_similarities, best_matches = process_entity(entity, max_tokens)
        
        queries_and_answers_text = f"Generated 5W1H Queries and Answers for '{entity}' (max tokens: {max_tokens}):\n\n"
        for query_key, query_data in queries_data.items():
            answer_key = f"answer_{query_key.split('_')[1]}"
            queries_and_answers_text += f"{query_key}: {query_data['text']}\n"
            queries_and_answers_text += f"{answer_key}: {corpus_data[answer_key]['text']}\n\n"
        
        embedding_summary = f"\nEmbedding-based Similarities ({EMBEDDING_MODEL}):\n"
        for query_key, query_similarities in cosine_similarities.items():
            embedding_summary += f"{query_key}:\n"
            for answer_key, similarity in query_similarities.items():
                embedding_summary += f"  {answer_key}: {similarity:.4f}\n"
            embedding_summary += "\n"
        
        best_matches_summary = "\nBest Matches:\n"
        for query_key, match_data in best_matches[EMBEDDING_MODEL].items():
            best_matches_summary += f"{query_key}:\n"
            best_matches_summary += f"  Query: {match_data['query_text']}\n"
            best_matches_summary += f"  Best Answer ({match_data['best_answer_key']}): {match_data['best_answer_text']}\n"
            best_matches_summary += f"  Cosine Similarity: {match_data['cosine_similarity']:.4f}\n\n"

        logging.info("Entity processing completed successfully")
        return queries_and_answers_text, embedding_summary, best_matches_summary, json.dumps({"entity": entity, "embedding_results": cosine_similarities, "best_matches": best_matches}, indent=2)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return error_message, "", "", "{}"

iface = gr.Interface(
    fn=integrated_interface,
    inputs=[
        gr.Textbox(label="Enter an entity (person, place, event, etc.)"),
        gr.Slider(minimum=50, maximum=2000, step=10, label="Max Tokens for Answer", value=100)
    ],
    outputs=[
        gr.Textbox(label="Generated Queries and Answers"),
        gr.Textbox(label="Embedding-based Similarities Summary"),
        gr.Textbox(label="Best Matches Summary"),
        gr.JSON(label="Detailed Results")
    ],
    title=f"Integrated 5W1H RAG Model with {EMBEDDING_MODEL} and Best Matches",
    description="Enter an entity and set the maximum tokens for answers. This will generate 5W1H queries, answers, calculate similarities using the embedding model, and find the best matches based on cosine similarity."
)

if __name__ == "__main__":
    iface.launch(debug=True)