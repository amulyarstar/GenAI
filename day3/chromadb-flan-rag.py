import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")

print("\nLoading FLAN model...")
# FLAN-T5 is a sequence-to-sequence model; use text2text-generation
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    device=-1
)
print("FLAN model loaded successfully!")

client = chromadb.PersistentClient(path="./university_rag_database")

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

print("\nCreating doc embeddings...")
document_embeddings = embedding_model.encode(documents)
print("Doc embeddings created:", document_embeddings.shape)

collection.upsert(
    ids=document_ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=document_embeddings.tolist()
)

def retrieve_documents(query: str, top_k: int = 2):
    query_embedding = embedding_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    return results

def create_prompt(query: str, retrieved_documents: list[str]) -> str:
    context = "\n".join(retrieved_documents)
    prompt = f"""Answer the question using only the context below. If the context does not contain the answer, say: I do not have enough information to answer your question.

Context:
{context}

Question:
{query}

Answer:"""
    return prompt

def rag_chatbot(query: str, top_k: int = 3):
    results = retrieve_documents(query, top_k=top_k)

    retrieved_documents = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    prompt = create_prompt(query, retrieved_documents)
    print("\nAugmented prompt:\n" + "-" * 30)
    print(prompt)

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    answer = result[0]["generated_text"]

    print("\nAnswer:\n" + "-" * 30)
    print(answer)
    print("=" * 60)

    return {
        "answer": answer,
        "retrieved_documents": retrieved_documents,
        "retrieved_metadatas": retrieved_metadatas,
        "distances": retrieved_distances
    }

# Test run
if __name__ == "__main__":
    response = rag_chatbot("Who can apply for scholarships?", top_k=2)
