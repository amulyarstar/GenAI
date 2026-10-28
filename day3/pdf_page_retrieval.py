!pip install -q pypdf
from pypdf import PdfReader

# ==========================================
# 1. LOAD PDF WITHOUT CHUNKING (Page = Document)
# ==========================================
pdf_path = "CA-1-CSE2266_TOC_ANNOUNCEMENT.pdf"  # Replace with your actual PDF path
reader = PdfReader(pdf_path)

documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        documents.append({
            "text": text,
            "score": 0.0,
            "page_number": i + 1
        })

print(f"Loaded {len(documents)} pages from the PDF as individual documents (No Chunking).")


# ==========================================
# 2. RETRIEVAL FUNCTION (Simple search over pages)
# ==========================================
def retrieve_documents(query, top_k=2):
    # For a real vector search, you would compare query embeddings with page embeddings here.
    # For demonstration without chunking, we return the top_k pages directly from our list.
    # (You can plug your embedding vector similarity score calculation here)
    for doc in documents:
        doc["score"] = 0.95  # Placeholder score
    return documents[:top_k]


# ==========================================
# 3. PROMPT CREATION
# ==========================================
def create_prompt(query, retrieved_documents):
    context = "\n".join(
        f"[Page {item.get('page_number', 'N/A')}]: {item['text']}"
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


# ==========================================
# 4. GENERATOR / ANSWER PIPELINE
# ==========================================
def answer_query(query, retrieve_func, generator_func):
    retrieved_documents = retrieve_func(
        query,
        top_k=2
    )

    print("\nRetrieved Pages:")
    for i, item in enumerate(retrieved_documents):
        print(f"Document {i+1} (Page {item.get('page_number')}):")
        print(f"Text Preview: {item['text'][:150]}...")
        print(f"Score: {item['score']}")

    prompt = create_prompt(
        query,
        retrieved_documents
    )

    print("\nAugmented prompt:")
    print(prompt)

    # Simulating generator call (Replace with your actual LLM generator pipeline)
    result = generator_func(prompt)

    print("=" * 60)
    print(result)

    return result


# Mock generator function for testing (Replace with your Hugging Face or OpenAI generator)
def mock_generator(prompt):
    return [{"generated_text": "Based on the PDF pages provided, here is the answer to your question."}]


# ==========================================
# 5. EXECUTION ENTRY POINT
# ========================================== 
user_question = input("Your question: ")

# 1. Check if input is empty
if not user_question.strip():
    print("Please enter a valid question.")

# 2. Check if user typed exit
elif user_question.strip().lower() == "exit":
    print("Thank you!")

# 3. Otherwise, run the pipeline once
else:
    answer_query(user_question, retrieve_documents, mock_generator)
