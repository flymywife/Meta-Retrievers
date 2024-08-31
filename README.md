# アプリの説明

## [app-historical-3.py]
Six-DimensionalQueryMeta-Retrievers

与えられたエンティティ（固有名詞）に対して話題の異なる6つの質問を生成します。
その後、そのクエリに対応するアンサーを生成します。
さらにクエリとアンサーをベクトル化してコサイン類似度またはBM25のスコアを計算して最適なマッチングを抽出します。

出力は以下の通りです：
- coupus_{date}_{maxToken}_{tempareture}.json: 生成した6つのコーパスをJSON出力します
- queries_{date}_{maxToken}_{tempareture}.json: 生成した6つのクエリをJSON出力します
- best_matches_{date}_{maxToken}_{tempareture}.json: クエリに対して最もコサイン類似度またはスコアの数値の高かった組み合わせをJSON出力します
- scores_{date}_{maxToken}_{tempareture}.json: クエリに対して全てのコーパスのスコアの数値をJSON出力します(BM25)
- cosine_similarities_{date}_{maxToken}_{tempareture}.json: クエリに対して全てのコーパスのコサイン類似度の数値をJSON出力します(ベクトル検索)
- vectors_{date}_{maxToken}_{tempareture}.json: テキストに対応したベクトルをJSON出力します

※固有名詞はある程度の年齢以上の有名人とかがいいです

コーパスのmaxTokensの長さを100-2000まで指定することができます。
100を超えるmaxTokensを指定すると100Tokensスタートで25Tokens刻みで自動的にテストします。

temparatureの指定ができます。

3つのEmbeddingモデル（text-embedding-3-large、text-embedding-3-small、text—embedding-ada-002）のコサイン類似度とBM25のスコアを出力します。

## [app-6entity.py]
SixTypesOfProperNounMeta-Retrievers

Six-DimensionalQueryMeta-Retrieversとの差分は以下の通りです：
- 固有名詞を最大6つ入力します。
- その後、それぞれの固有名詞に対応した自由なクエリを１つずつ生成します。

## [analyze-app.py]
Meta-Retrievers分析ツール
Six-DimensionalQueryMeta-RetrieversとSixTypesOfProperNounMeta-Retrieversの
試験結果であるbest_matches_{date}_{maxToken}_{tempareture}.jsonをエンティティとモデルごとにアップすることで
エラーの数の分析ができます。
出力はHTMLファイルになります。
Retrieverは手動入力です。
まとめてアップするときに読み取りたいトークンの範囲を設定できます。
ファイル名の最初が「best_matches」になっているファイルだけ読み取ります。

## [analyze-total-app.py]
Meta-Retrievers総合分析ツール
Meta-Retrievers分析ツールを用いて出力したHTMLファイルをまとめてアップすると
RetrieverとtemparatureごとにエラーカウントしたHTMLファイルが出力できます。
量が多くなりすぎて計算が難しくなってきた時に使ってください。

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
   python app-historical-3.py
   ```

2. Webブラウザを開き、コンソールに表示されるURL（通常はhttp://127.0.0.1:7860）にアクセスします。

3. エンティティ入力フィールドに、固有名詞を入力します。

4. maxTokenを指定してください。

5. temparatureを指定してください

6. "Submit"ボタンを押すと動き出します。