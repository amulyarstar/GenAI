"""
04_text_chunking_strategies.py
==============================
Stage 4: Chunking Strategies for RAG.
Demonstrates:
  1. Fixed-size character chunking
  2. Sliding window character chunking with overlap
  3. LangChain RecursiveCharacterTextSplitter
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Raw documents joined into a coherent continuous text corpus
raw_documents = [
    "Students can apply for merit and need-based scholarships through the student portal.",
    "Java remains an evergreen programming language widely used in enterprise backend systems.",
    "The university library is open 24/7 during semester final examinations.",
    "Machine learning algorithms allow computers to discover patterns directly from training data.",
    "Students must complete all prerequisite course requirements before registering for electives.",
    "The university examination department releases hall tickets two weeks prior to assessments.",
    "Internship opportunities require maintaining a cumulative CGPA of 7.5 or higher.",
]

full_text = " ".join(raw_documents)
print(f"Total Source Text Length: {len(full_text)} characters\n")

# =====================================================================
# Strategy 1: Fixed Character Chunking (No Overlap)
# =====================================================================
chunk_size = 120
fixed_chunks = []

for i in range(0, len(full_text), chunk_size):
    fixed_chunks.append(full_text[i : i + chunk_size])

print("=" * 60)
print(f"1. FIXED CHARACTER CHUNKING (Size: {chunk_size}, Total: {len(fixed_chunks)})")
print("=" * 60)
for i, chunk in enumerate(fixed_chunks, start=1):
    print(f"[Chunk {i}] ({len(chunk)} chars): {chunk}")


# =====================================================================
# Strategy 2: Sliding Window Chunking (With Overlap)
# =====================================================================
chunk_overlap = 30
step = chunk_size - chunk_overlap
overlap_chunks = []

for i in range(0, len(full_text), step):
    overlap_chunks.append(full_text[i : i + chunk_size])

print("\n" + "=" * 60)
print(f"2. SLIDING WINDOW WITH OVERLAP (Size: {chunk_size}, Overlap: {chunk_overlap}, Total: {len(overlap_chunks)})")
print("=" * 60)
for i, chunk in enumerate(overlap_chunks, start=1):
    print(f"[Chunk {i}] ({len(chunk)} chars): {chunk}")


# =====================================================================
# Strategy 3: LangChain Recursive Character Text Splitter
# =====================================================================
# Respects semantic paragraph, sentence, and word boundaries
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

recursive_chunks = recursive_splitter.split_text(full_text)

print("\n" + "=" * 60)
print(f"3. RECURSIVE CHARACTER SPLITTER (Total Chunks: {len(recursive_chunks)})")
print("=" * 60)
for i, chunk in enumerate(recursive_chunks, start=1):
    print(f"[Chunk {i}] ({len(chunk)} chars): {chunk}")
