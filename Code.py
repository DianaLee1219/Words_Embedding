import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding

# 샘플 데이터: 간단한 문장들의 모음
sentences = [
    "I love machine learning",
    "I love coding in Python",
    "Deep learning is fun"
]

# 각 문장을 단어로 분할하고, 각 단어에 대한 고유한 인덱스를 생성
word_index = {}

for sentence in sentences:
    for word in sentence.split():
        if word not in word_index:
            word_index[word] = len(word_index) + 1
            
word_index

# 문장들을 단어 인덱스의 시퀀스로 변환
sequences = [[word_index[word] for word in sentence.split()] for sentence in sentences]
sequences

# 문장들 중 가장 긴 것의 길이를 구함
max_length = max([len(seq) for seq in sequences])

# 모든 문장을 가장 긴 문장의 길이로 패딩
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
padded_sequences

# Embedding 레이어 생성
embedding_dim = 8
embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_length)

# 패딩된 시퀀스를 Embedding 레이어에 통과시켜 임베딩된 결과를 얻음
embedded_sequences = embedding_layer(padded_sequences)

print(embedded_sequences.shape)
print(embedded_sequences)

# Embedding 레이어의 가중치 (단어 임베딩 행렬) 출력
embeddings = embedding_layer.get_weights()[0]
print("Embedding Layer Shape :", embeddings.shape)
print("Embedding Layer Weights (Word Embeddings):\n", embeddings)
print()

# 예: 'love'라는 단어의 임베딩 벡터를 출력
print("\nEmbedding for 'love':\n", embeddings[word_index['love']])
