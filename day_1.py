import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Ensure the model is loaded first
print("Loading model...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    token=False
)
print("Model loaded!")

# 2. Extract internal components
sentence = "I am AMULYA"
tokenizer = model[0].tokenizer
minilm = model[0].auto_model

# 3. Tokenize and extract embeddings
inputs = tokenizer(sentence, return_tensors="pt")
print("Token IDs:", inputs["input_ids"])

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)

minilm = minilm.to("cpu")
X = minilm.get_input_embeddings()(inputs["input_ids"])
X = X[0].detach().numpy()

print("Token embedding shape (Tokens x Dimensions):", X.shape)

Q = X
K = X
V = X
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)


scores = (Q @ K.T) / np.sqrt(K.shape[1])
print("Scores:\n", scores.round(2))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

attention = softmax(scores, axis=1)
print("Attention Matrix:\n", attention.round(2))

output = attention @ V
print("Attention output shape:", output.shape)
print("First token output (first 10 dims):", output[0][:10].round(3))

# 4. Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

plt.imshow(attention, cmap="Oranges")
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Key tokens")
plt.ylabel("Query tokens")
plt.title("Simplified Attention Using MiniLM Embeddings")
plt.colorbar()
plt.tight_layout()
plt.show()
