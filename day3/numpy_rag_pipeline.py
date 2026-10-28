import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# 1. Load the embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# 2. Load the language model
print("\nLoading language model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

print("Language model loaded successfully!")


# 3. Sample documents
documents = [
    "Students can apply for scholarships.",
    "Java is an evergreen programming language.",
    "The university library is open now.",
    "Students can run programs.",
    "Machine learning allows computers to learn from data.",
    "Students should complete all requirements on time.",
    "Students must complete their examinations.",
    "The university provides examinations."
]


# 4. Generate document embeddings
print("\nCreating document embeddings...")

document_embeddings = model.encode(documents)

print("Embedding shape:", document_embeddings.shape)


# 5. Retrieve relevant documents
def retrieve_documents(query, top_k=2):

    query_embedding = model.encode([query])

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    retrieved_documents = []

    for index in top_indices:
        retrieved_documents.append({
            "document": documents[index],
            "score": float(similarity_scores[index])
        })

    return retrieved_documents


# 6. Create the augmented prompt
def create_prompt(query, retrieved_documents):

    context = "\n".join(
        item["document"]
        for item in retrieved_documents
    )

    prompt = f"""
Answer the question using only the context below.

If the context does not contain the answer, say:
I do not have enough information to answer your question.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


# 7. Generate the answer
def answer_query(query):

    retrieved_documents = retrieve_documents(
        query,
        top_k=2
    )

    print("\nRetrieved Documents:")

    for i, item in enumerate(retrieved_documents, start=1):
        print(f"\nDocument {i}:")
        print("Text:", item["document"])
        print("Similarity:", round(item["score"], 4))

    prompt = create_prompt(
        query,
        retrieved_documents
    )

    print("\nAugmented Prompt:")
    print(prompt)

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    answer = result[0]["generated_text"]

    print("\nGenerated Answer:")
    print(answer)

    print("=" * 60)

    return answer


# 8. Main interaction loop
while True:

    user_question = input(
        "\nYour question (or type 'exit' to stop): "
    ).strip()

    if user_question.lower() == "exit":
        print("Thank you!")
        break

    if not user_question:
        print("Please enter a valid question.")
        continue

    answer_query(user_question)
