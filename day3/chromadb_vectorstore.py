import chromadb
from sentence_transformers import SentenceTransformer

print("Loading the model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# Initialize ChromaDB persistent storage
client = chromadb.PersistentClient(path="./university_chroma_db")

collection = client.get_or_create_collection(
    name="university_collection",
    metadata={"hnsw:space": "cosine"}
)
print("Collection ready.")

documents = [
    "students can apply scholarship",
    "java is a evergreen pl",
    "the uni li is open now",
    "students can run ",
    "ml allows computers to learn from data",
    "students are good they complete every req fastt",
    "students complete every exams",
    "the univerrsity provides examinations",
    "students can apply scholarship",
    "java is a evergreen pl",
    "the uni li is open now",
    "students can run ",
    "ml allows computers to learn from data"
]

document_ids = [f"doc{i+1}" for i in range(len(documents))]

metadatas = [
    {"topic": "scholarship", "source": "university_handbook"},
    {"topic": "java", "source": "internship_policy"},
    {"topic": "Library", "source": "university_handbook"},
    {"topic": "internship", "source": "university_rules"},
    {"topic": "scholarship", "source": "university_rules"},
    {"topic": "career", "source": "university_rules"},
    {"topic": "academics", "source": "university_rules"},
    {"topic": "exams", "source": "university_rules"},
    {"topic": "scholarship", "source": "university_handbook"},
    {"topic": "java", "source": "internship_policy"},
    {"topic": "Library", "source": "university_handbook"},
    {"topic": "internship", "source": "university_rules"},
    {"topic": "scholarship", "source": "university_rules"}
]

print("\nCreating document embeddings...")
document_embeddings = embedding_model.encode(documents)
print("Doc embeddings created:", document_embeddings.shape)

collection.upsert(
    ids=document_ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=document_embeddings.tolist()
)

def search_documents(query: str, top_k: int = 2):
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    return results

while True:
    user_question = input("\nYour Question: ")

    if user_question.lower() == "exit":
        print("\nThank Amulya for the bot.")
        break

    if not user_question.strip():
        print("\nPlease enter a valid question.")
        continue

    print("\nUser Question:", user_question)

    results = search_documents(user_question, top_k=2)

    retrieved_documents = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    for i, document in enumerate(retrieved_documents):
        distance = retrieved_distances[i]
        similarity = 1 - distance
        metadata = retrieved_metadatas[i]

        print(f"\nDocument {i+1}:")
        print("Content:   ", document)
        print("Topic:     ", metadata.get("topic", "N/A"))
        print("Source:    ", metadata.get("source", "N/A"))
        print(f"Similarity: {similarity:.4f}")
        print(f"Metadata:   {metadata}")
