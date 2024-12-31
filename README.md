#Text Embedding

corpus = df['processed_text'].tolist()
word2vec_params = {
    'vector_size': [100, 200, 300],
    'window': [5, 7, 9],
    'min_count': [1, 3, 5]
}

model = Word2Vec(corpus, vector_size=200, window=7, min_count=3, workers=4, sg=1)
