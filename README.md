
# News Headline Categorization using SentenceTransformers

This project demonstrates how to categorize news headlines by calculating their similarity to existing headlines using a pre-trained model from the `sentence_transformers` library. Specifically, the `all-MiniLM-L6-v2` model is used to generate embeddings for the headlines, and cosine similarity is computed to find the closest matching category.

## Project Overview

The notebook performs the following steps:

1. **Importing Necessary Libraries:**
   - `SentenceTransformer` and `util` from the `sentence_transformers` module for text embedding and similarity calculations.
   - `pandas` for handling the news headline dataset.

2. **Loading the Model:**
   - The `all-MiniLM-L6-v2` model is loaded to generate embeddings for the news headlines.

3. **Reading the Dataset:**
   - A dataset containing news headlines and their respective categories is loaded from a CSV file (`news_headlines.csv`) into a Pandas DataFrame.

4. **Tokenizing and Encoding Headlines:**
   - The news headlines are converted into embeddings using the loaded model.

5. **Calculating Similarity:**
   - A new headline is introduced, and its embedding is calculated.
   - Cosine similarity is computed between the new headline's embedding and the embeddings of the existing headlines.

6. **Ranking and Categorizing:**
   - The existing headlines are ranked based on their similarity to the new headline.
   - The category of the most similar headlines is used to infer the category of the new headline.

## Key Code Snippets

- **Importing Libraries:**
    ```python
    from sentence_transformers import SentenceTransformer, util
    import pandas as pd
    ```

- **Loading the Model and Dataset:**
    ```python
    model = SentenceTransformer('all-MiniLM-L6-v2')
    news_headlines_df = pd.read_csv("Resources/news_headlines.csv")
    ```

- **Generating Embeddings:**
    ```python
    news_headlines_embeddings = model.encode(news_headlines)
    new_headline_embedding = model.encode([new_headline])
    ```

- **Calculating Similarity:**
    ```python
    cosine_similarity_score = util.cos_sim(headline_embedding, new_headline_embedding)
    ```

- **Sorting and Ranking:**
    ```python
    similarities.sort(key=lambda x: x[1], reverse=True)
    ```

## Output

The notebook provides the following output:

- The new headline to categorize.
- A ranked list of existing headlines based on similarity scores, including their categories.
- The similarity scores for each headline.

### Example Output

```text
News headline to categorize: Top 10 Hacks for Traveling Like a Pro.

Rank 1: Category: Technology, Headline: Hacker Pleads Guilty To Stealing Over 100,000 Passwords for Reddit
Similarity score: 0.3084

Rank 2: Category: Travel, Headline: The 5 Best Restaurants In The World
Similarity score: 0.2657

Rank 3: Category: Travel, Headline: The Best Sub Shops in the Caribbean You Should Visit This Summer
Similarity score: 0.2085
...
```

## Installation

To run this notebook, you need to have Python installed along with the necessary packages. You can install the required libraries using pip:

```bash
pip install sentence-transformers pandas
```

## Usage

Clone the repository, navigate to the project directory, and open the Jupyter notebook. Execute the cells to see the results.

```bash
git clone <repository-url>
cd <repository-folder>
jupyter notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
