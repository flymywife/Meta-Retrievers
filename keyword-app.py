import gradio as gr
import json
from rank_bm25 import BM25Okapi
import re
import os

def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def read_json_file(file):
    try:
        print(f"Attempting to read file: {file.name}")
        print(f"File object type: {type(file)}")
        print(f"File object attributes: {dir(file)}")
        
        # Check if the file exists and is readable
        if not os.path.exists(file.name):
            raise FileNotFoundError(f"The file '{file.name}' does not exist.")
        if not os.access(file.name, os.R_OK):
            raise PermissionError(f"No read permission for file '{file.name}'.")
        
        with open(file.name, 'rb') as f:
            file_content = f.read()
        
        print(f"Successfully read {len(file_content)} bytes from the file.")
        
        if not file_content:
            raise ValueError(f"The file '{file.name}' is empty.")
        
        # Try UTF-8 first
        try:
            decoded_content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying with 'latin-1'")
            # If UTF-8 fails, try with 'latin-1'
            decoded_content = file_content.decode('latin-1')
        
        print(f"Successfully decoded the file content.")
        
        json_data = json.loads(decoded_content)
        print("Successfully parsed JSON data.")
        return json_data
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        raise ValueError(f"Invalid JSON format in file '{file.name}': {str(e)}")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise ValueError(f"Error reading file '{file.name}': {str(e)}")

def match_queries_to_answers(queries_file, corpus_file):
    try:
        # Read and parse JSON files
        queries = read_json_file(queries_file)
        corpus = read_json_file(corpus_file)

        # Validate JSON structure
        if 'entity' not in queries or 'queries' not in queries:
            raise ValueError("Invalid queries JSON structure. 'entity' and 'queries' fields are required.")
        if 'entity' not in corpus or 'corpus' not in corpus:
            raise ValueError("Invalid corpus JSON structure. 'entity' and 'corpus' fields are required.")

        # Extract query texts and answer texts
        query_texts = [item['text'] for item in queries['queries'].values()]
        answer_texts = [item['text'] for item in corpus['corpus'].values()]

        # Preprocess texts
        preprocessed_queries = [preprocess_text(text) for text in query_texts]
        preprocessed_answers = [preprocess_text(text) for text in answer_texts]

        # Create BM25 model using the answers
        bm25 = BM25Okapi(preprocessed_answers)

        # Match queries to answers
        matches = []
        for query, orig_query in zip(preprocessed_queries, query_texts):
            scores = bm25.get_scores(query.split())
            best_match_index = scores.argmax()
            matches.append({
                "query": orig_query,
                "best_match_answer": answer_texts[best_match_index],
                "score": float(scores[best_match_index])
            })

        # Sort matches by score
        matches.sort(key=lambda x: x['score'], reverse=True)

        # Prepare output JSON
        output = {
            "entity": queries["entity"],
            "matches": matches
        }

        return json.dumps(output, indent=2)
    except ValueError as e:
        print(f"ValueError occurred: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}, indent=2)

# Define Gradio interface
iface = gr.Interface(
    fn=match_queries_to_answers,
    inputs=[
        gr.File(label="Queries JSON"),
        gr.File(label="Corpus (Answers) JSON")
    ],
    outputs=gr.JSON(label="Query-Answer Matches"),
    title="Query-Answer Matching Tool",
    description="Upload queries JSON and corpus (answers) JSON files to match each query with the most relevant answer."
)

# Launch the interface
iface.launch(debug=True)