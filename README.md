# RAG Gradio Application for Entity Analysis

This application is a Retrieval-Augmented Generation (RAG) system that generates questions and answers about a given entity using OpenAI's GPT model, calculates similarities between queries and answers, and finds the best matching pairs.

## Features

- Generates 5W1H (Who, What, When, Where, Why, How) questions about a given entity
- Produces answers to these questions
- Calculates cosine similarities between queries and answers
- Finds the best matching answer for each query based on similarity
- Allows selection between different embedding models
- Stores results in JSON files with timestamps for historical tracking
- Provides a user-friendly interface using Gradio

## Requirements

- Python 3.7+
- OpenAI API key

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your OpenAI API key:
   Create a `.env` file in the project root and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to the URL displayed in the console (usually `http://127.0.0.1:7860`).

3. In the text input field, enter the name of an entity (person, place, event, etc.) you want to analyze.

4. Select the embedding model you want to use from the dropdown menu.

5. Click the "Submit" button.

6. The application will generate:
   - 5W1H questions about the entity
   - Answers to these questions
   - Cosine similarities between queries and answers
   - Best matching query-answer pairs based on similarity

## Output

The application generates four types of output:

1. Generated Queries and Answers: Displayed in the first text box of the interface.
2. Cosine Similarities Summary: Shown in the second text box.
3. Detailed Similarities: Presented in JSON format in the third box.
4. Best Matches Summary: Displayed in the fourth text box, showing the best matching query-answer pairs.

Additionally, four JSON files are saved in the `history/{model_name}` folder for each run:
- `queries_YYYYMMDD_HHMMSS.json`: Contains the generated queries and their vectors.
- `corpus_YYYYMMDD_HHMMSS.json`: Contains the generated answers and their vectors.
- `similarities_YYYYMMDD_HHMMSS.json`: Contains the calculated similarities between queries and answers.
- `best_matches_YYYYMMDD_HHMMSS.json`: Contains the best matching query-answer pairs based on similarity.

## Embedding Models

The application supports two embedding models:
1. text-embedding-ada-002
2. text-embedding-3-small

You can select the desired model from the dropdown menu in the user interface. The chosen model will be used for vectorizing the queries and answers, which affects the similarity calculations and best match results.

## Note

This application uses the OpenAI API, which may incur costs. Please be aware of your usage and any associated fees.

## Contributing

Contributions to improve the application are welcome. Please feel free to submit a Pull Request.

