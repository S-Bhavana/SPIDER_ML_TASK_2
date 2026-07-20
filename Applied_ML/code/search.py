import faiss
import pickle
from sentence_transformers import SentenceTransformer


INDEX_PATH = "outputs/medquad.index"
DATA_PATH = "outputs/medquad_data.pkl"


print("Loading index...")

index = faiss.read_index(INDEX_PATH)

with open(DATA_PATH, "rb") as f:
    texts = pickle.load(f)


print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


print("Medical QA System Ready!")


while True:

    query = input("\nEnter your question (type exit to stop): ")

    if query.lower() == "exit":
        break

    query_vector = model.encode([query]).astype("float32")

    distances, results = index.search(query_vector, 3)

    print("\nTop Answers:\n")

    for idx in results[0]:
        print(texts[idx])
        print("-" * 50)