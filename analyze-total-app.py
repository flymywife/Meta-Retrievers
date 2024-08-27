import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
from datetime import datetime

def extract_model_name(filename):
    return filename.split('_')[0]

def parse_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        rows = soup.find_all('tr')
        data = []
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) == 4:
                _, max_tokens, temperature, error_count = [col.text.strip() for col in cols]
                error_count = 0 if error_count == 'No errors' else int(error_count)
                data.append({
                    'Max Tokens': int(max_tokens),
                    'Temperature': float(temperature),
                    'Error Count': error_count
                })
    return pd.DataFrame(data)

def analyze_files(file_paths):
    results_by_tokens = defaultdict(lambda: defaultdict(int))
    results_by_temperature = defaultdict(lambda: defaultdict(int))
    
    for file_path in file_paths:
        model_name = extract_model_name(os.path.basename(file_path))
        df = parse_html_file(file_path)
        
        # Analyze by Max Tokens
        for max_tokens, group in df.groupby('Max Tokens'):
            results_by_tokens[model_name][max_tokens] += group['Error Count'].sum()
        
        # Analyze by Temperature
        for temperature, group in df.groupby('Temperature'):
            results_by_temperature[model_name][temperature] += group['Error Count'].sum()
    
    return results_by_tokens, results_by_temperature

def generate_html_output(results_by_tokens, results_by_temperature):
    html = '''
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1, h2 { color: #333; }
            .model-container { margin-bottom: 40px; }
            .table-container { display: flex; gap: 20px; flex-wrap: wrap; }
            .table-wrapper { flex: 1; min-width: 300px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
    </head>
    <body>
    <h1>Error Count Analysis</h1>
    '''
    for model in results_by_tokens.keys():
        html += f'<div class="model-container"><h2>{model}</h2>'
        html += '<div class="table-container">'
        
        # Table for Max Tokens
        html += '<div class="table-wrapper"><h3>By Max Tokens</h3>'
        html += '<table><tr><th>Max Tokens</th><th>Total Error Count</th></tr>'
        for max_tokens, error_count in sorted(results_by_tokens[model].items()):
            html += f'<tr><td>{max_tokens}</td><td>{error_count}</td></tr>'
        html += '</table></div>'
        
        # Table for Temperature
        html += '<div class="table-wrapper"><h3>By Temperature</h3>'
        html += '<table><tr><th>Temperature</th><th>Total Error Count</th></tr>'
        for temperature, error_count in sorted(results_by_temperature[model].items()):
            html += f'<tr><td>{temperature}</td><td>{error_count}</td></tr>'
        html += '</table></div>'
        
        html += '</div></div>'
    html += '</body></html>'
    return html

def process_files(files):
    if not files:
        return "No files uploaded. Please upload HTML files for analysis."
    
    file_paths = [file.name for file in files]
    results_by_tokens, results_by_temperature = analyze_files(file_paths)
    html_output = generate_html_output(results_by_tokens, results_by_temperature)
    
    # Generate filename with current date and time
    now = datetime.now()
    filename = f"total_analyze_{now.strftime('%Y%m%d_%H%M%S_%f')}.html"
    
    # Save HTML to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    return html_output, filename

iface = gr.Interface(
    fn=process_files,
    inputs=gr.File(file_count="multiple"),
    outputs=[gr.HTML(), gr.File(label="Download Analysis Result")],
    title="Embedding Model Error Analysis",
    description="Upload HTML files containing embedding model error data for analysis."
)

if __name__ == "__main__":
    iface.launch(share=True)