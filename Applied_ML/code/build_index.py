import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os


DATA_PATH = "dataset/medquad.csv"
INDEX_PATH = "outputs/medquad.index"
DATA_STORE_PATH = "outputs/medquad_data.pkl"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print(df.head())


texts = []

for _, row in df.iterrows():
    text = str(row["question"]) + " " + str(row["answer"])
    texts.append(text)


print("Total documents:", len(texts))


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")


print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


faiss.write_index(index, INDEX_PATH)


with open(DATA_STORE_PATH, "wb") as f:
    pickle.dump(texts, f)


print("Index created successfully!")