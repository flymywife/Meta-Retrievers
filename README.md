#アプリの説明

## [app.py]
Six-Dimensional Query Tool
与えられた「固有名詞」に対して
Six-Dimensional Query(Who, What, When, Where, Why, How)を生成します。
その後、そのクエリに対応するアンサーを生成します。
さらにクエリとアンサーをベクトル化して
コサイン類似度またはBM25のスコアを計算して最適なマッチングを抽出します。
出力は
coupus_date_maxToken.json:生成した6つのコーパスをJSON出力します
queries_date_maxToken.json:生成した6つのクエリをJSON出力します
best_matches_date_maxToken.json:クエリに対して最もコサイン類似度またはスコアの数値の高かった組み合わせをJSON出力します
scores_date_maxToken.json:クエリに対して全てのコーパスのスコアの数値をJSON出力します(BM25)
cosine_similarities_date_maxToken.json:クエリに対して全てのコーパスのコサイン類似度の数値をJSON出力します(ベクトル検索)
vectors_date_maxToken.json:テキストに対応したベクトルをJSON出力します
※固有名詞はポピュラーな歴史上の人物とかが良いです
アンサーのトークンの最大の長さを50-500まで指定することができます。
3つのEmbeddingモデル（text-embedding-3-largeとtext-embedding-3-smallとtext—embedding-ada-002）
のコサイン類似度とBM25のスコアを出力します。

## [app-large.py]
上記Six-Dimensional Query Toolの追加実験用です。
text-embedding-3-largeモデルの実験をmaxToken2000まで実施できます。

## [app-6entity.py]
Proper Noun Tool
固有名詞を最大6つ入力します
その後、それぞれの固有名詞に対応した自由なクエリを１つずつ生成します
さらにクエリとアンサーをベクトル化して
コサイン類似度またはBM25のスコアを計算して最適なマッチングを抽出します。

## 必要条件
- Python 3.10+
- OpenAI API key

## インストール
1. 必要なパッケージをインストールします：
   ```
   pip install -r requirements.txt
   ```

2. OpenAI APIキーを設定します
プロジェクトのルートディレクトリに.envファイルを作成し、APIキーを追加します：
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 使用方法

1. アプリケーションを実行します：
   ```
   python app.py
   ```

2. Webブラウザを開き、コンソールに表示されるURL（通常はhttp://127.0.0.1:7860）にアクセスします。

3. テキスト入力フィールドに、固有名詞を入力します。

4. maxTokenを指定してください。

5. "Submit"ボタンを押すと動き出します。




## [app.py]
Six-Dimensional Query Tool
This tool generates a Six-Dimensional Query (Who, What, When, Where, Why, How) for a given "proper noun".
It then generates answers corresponding to these queries.
Furthermore, it vectorizes the queries and answers to calculate cosine similarity or BM25 scores to extract the optimal matching.
The outputs are:
coupus_date_maxToken.json: Outputs the 6 generated corpora in JSON format
queries_date_maxToken.json: Outputs the 6 generated queries in JSON format
best_matches_date_maxToken.json: Outputs the combinations with the highest cosine similarity or score for each query in JSON format
scores_date_maxToken.json: Outputs the scores for all corpora against each query in JSON format (BM25)
cosine_similarities_date_maxToken.json: Outputs the cosine similarity values for all corpora against each query in JSON format (Vector search)
vectors_date_maxToken.json: Outputs the vectors corresponding to the texts in JSON format
Note: Popular historical figures are good choices for proper nouns.
The maximum length of answer tokens can be specified from 50 to 500.
It outputs cosine similarities and BM25 scores for three Embedding models (text-embedding-3-large, text-embedding-3-small, and text-embedding-ada-002).

## [app-large.py]
This is for additional experiments with the above Six-Dimensional Query Tool.
It can conduct experiments with the text-embedding-3-large model up to maxToken 2000.

## [app-6entity.py]
Proper Noun Tool
Input up to 6 proper nouns.
Then, it generates one free query corresponding to each proper noun.
It further vectorizes the queries and answers to calculate cosine similarity or BM25 scores and extract the optimal matching.

## Requirements
Python 3.10+
OpenAI API key

## Installation

Install the required packages:
   ```
   pip install -r requirements.txt
   ```


Set up the OpenAI API key
Create a .env file in the project root directory and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
## Usage

Run the application:
   ```
   python app.py
   ```

Open a web browser and access the URL displayed in the console (usually http://127.0.0.1:7860).
Enter a proper noun in the text input field.
Specify the maxToken.
Press the "Submit" button to start the process.