import gradio as gr
import json
import os
from datetime import datetime
import re

def analyze_json_files(files, model_name):
    results = []
    for file in files:
        filename = file.name
        match = re.search(r'_(\d+)_(\d+\.\d+)\.json$', filename)
        if match:
            max_tokens = int(match.group(1))
            temperature = float(match.group(2))
        else:
            max_tokens = "N/A"
            temperature = "N/A"

        with open(file.name, 'r') as f:
            data = json.load(f)
        
        entity = data.get('entity', 'Unknown')
        error_count = 0

        for key, value in data.get('best_matches', {}).items():
            query_number = key.split('_')[-1]
            answer_key = value.get('best_answer_key', '')
            if f"answer_{query_number}" != answer_key:
                error_count += 1

        results.append({
            'Entity': entity,
            'Max Tokens': max_tokens,
            'Temperature': temperature,
            'Error Count': error_count
        })

    # Sort results by Temperature and Max Tokens in ascending order
    results.sort(key=lambda x: (x['Temperature'], x['Max Tokens']))

    # Generate HTML output
    html_output = "<table border='1'><tr><th>Entity</th><th>Max Tokens</th><th>Temperature</th><th>Error Count</th></tr>"
    for result in results:
        html_output += f"<tr><td>{result['Entity']}</td><td>{result['Max Tokens']}</td><td>{result['Temperature']}</td><td>{'No errors' if result['Error Count'] == 0 else result['Error Count']}</td></tr>"
    html_output += "</table>"

    # Generate output filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name}_{results[0]['Entity']}_{current_time}.html"

    # Save HTML to file
    with open(output_filename, 'w') as f:
        f.write(html_output)

    return html_output, output_filename

# Define Gradio interface
iface = gr.Interface(
    fn=analyze_json_files,
    inputs=[
        gr.File(file_count="multiple", label="Upload JSON files"),
        gr.Textbox(label="Model Name")
    ],
    outputs=[
        gr.HTML(label="Analysis Results"),
        gr.File(label="Download HTML Output")
    ],
    title="JSON File Analyzer",
    description="Upload multiple JSON files and analyze them based on the given criteria."
)

# Launch the interface
iface.launch()
