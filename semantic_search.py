from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")
print("embedded")

documents = [
    "students can apply scholarship",
    "java is a evergreen pl",
    "the uni lib is open now",
    "students can run ",
    "ml allows computers to learn from data"
]

query ="How can i get the financial support for education"
print("\n user q")
print(query)

document_embeddings = model.encode(documents)
print(document_embeddings.shape)

query_embedding = model.encode([query])
print(query_embedding.shape)

similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)

print(similarity_scores)

EEE
