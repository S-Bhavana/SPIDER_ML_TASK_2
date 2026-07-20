import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle


# Dataset path
DATA_PATH = "../dataset/medquad.csv"

# Output paths
INDEX_PATH = "../outputs/medquad.index"
DATA_STORE_PATH = "../outputs/medquad_data.pkl"


print("Loading dataset...")

# Read CSV
df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print(df.head())


# Combine question and answer text
texts = []

for i, row in df.iterrows():
    text = str(row["question"]) + " " + str(row["answer"])
    texts.append(text)


print("Total documents:", len(texts))


# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Create embeddings
print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)


embeddings = np.array(embeddings).astype("float32")


# Create FAISS index
print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# Save index
faiss.write_index(
    index,
    INDEX_PATH
)


# Save original text data
with open(DATA_STORE_PATH, "wb") as f:
    pickle.dump(texts, f)


print("Index created successfully!")
print("Saved at:", INDEX_PATH)