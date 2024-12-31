# Disaster-Tweet
#Text Embedding
# 2. Word2Vec Embedding with Hyperparameter Tuning
corpus = df['processed_text'].tolist()

# Hyperparameter tuning for Word2Vec (example)
word2vec_params = {
    'vector_size': [100, 200, 300],
    'window': [5, 7, 9],
    'min_count': [1, 3, 5]
}

# Use GridSearchCV to find the best hyperparameters for Word2Vec
# ... (Code for GridSearchCV with Word2Vec) ...

# After tuning, create the Word2Vec model with the best parameters
model = Word2Vec(corpus, vector_size=200, window=7, min_count=3, workers=4, sg=1)  # Replace with best parameters
