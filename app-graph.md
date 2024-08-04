graph TD
    A[Start] --> B[User Input: Entity & Max Tokens]
    B --> C[Generate 5W1H Queries]
    C --> D[Generate Answers]
    D --> E[Vectorize Queries and Answers]
    E --> F[Calculate Cosine Similarities]
    E --> G[Calculate BM25 Scores]
    F --> H[Find Best Matches]
    G --> H
    H --> I[Prepare Summaries]
    I --> J[Save Results as JSON]
    J --> K[Display Results in UI]
    K --> L[End]

    subgraph "Embedding Models"
    E1[text-embedding-ada-002]
    E2[text-embedding-3-small]
    E3[text-embedding-3-large]
    end

    E --> E1
    E --> E2
    E --> E3

    subgraph "Output"
    O1[Queries and Answers]
    O2[Embedding Similarities]
    O3[BM25 Scores]
    O4[Best Matches]
    O5[Detailed JSON Results]
    end

    K --> O1
    K --> O2
    K --> O3
    K --> O4
    K --> O5