from huggingface_hub import login
from getpass import getpass
import torch
from transformers import pipeline
import pandas as pd

# Step 1: Hugging Face authentication
token = getpass("Enter your Hugging Face token: ")
login(token=token)
del token

print("Hugging Face login successful!")

from transformers import pipeline
chatbot = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu"
)

print("Model Successfully")

# Load the e-commerce dataset if you haven't loaded it already
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "FloZewi/E-commerce-Data-Analysis/master/"
    "ecommerce_product_dataset.csv"
)

df = pd.read_csv(DATA_URL)
df.columns = df.columns.str.strip()

# System prompt for the e-commerce chatbot
messages = [
    {
        "role": "system",
        "content": (
            "You are ShopMate, a friendly, helpful and knowledgeable "
            "e-commerce shopping assistant. Answer questions about products, "
            "prices, ratings, discounts, stock availability and categories. "
            "Use the product dataset provided in the conversation to answer "
            "factual questions. Never invent product names, prices, ratings, "
            "discounts or stock information. If the required information is "
            "not provided, say that you cannot verify it from the dataset. "
            "Help users compare products and find suitable options. "
            "Keep answers clear, friendly and easy to understand."
        )
    },
]

print("Welcome to ShopMate - E-commerce Assistant! 🛍️")
print("Ask about products, prices, ratings, discounts or categories.")
print("Type 'exit' to stop the conversation.\n")


while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye! Happy shopping 🛒")
        break

    if not user_input:
        continue

    # Find relevant rows from the actual dataset
    searchable = df.astype(str).apply(
        lambda col: col.str.lower().str.contains(
            user_input.lower(),
            regex=False,
            na=False
        )
    )

    matching_rows = df[searchable.any(axis=1)].head(10)

    # Include dataset context with the user's question
    if not matching_rows.empty:
        dataset_context = matching_rows.to_string(index=False)
    else:
        # Provide a small overview when exact matches aren't found
        dataset_context = (
            "No exact product match was found. "
            "Dataset columns: " + ", ".join(df.columns) +
            "\nAvailable categories: " +
            ", ".join(df["Category"].dropna().unique().astype(str))
        )

    messages.append({
        "role": "user",
        "content": (
            f"User question: {user_input}\n\n"
            f"Relevant dataset information:\n{dataset_context}\n\n"
            "Answer using the supplied dataset information. "
            "Do not invent missing facts."
        )
    })

    try:
        response = chatbot(
            messages,
            max_new_tokens=200,
            do_sample=False,
            return_full_text=False
        )

        answer = response[0]["generated_text"]

        # Handle chat-formatted output if returned by the pipeline
        if isinstance(answer, list):
            assistant_messages = [
                item for item in answer
                if isinstance(item, dict)
                and item.get("role") == "assistant"
            ]
            if assistant_messages:
                answer = assistant_messages[-1].get("content", "")
            elif answer and isinstance(answer[-1], dict):
                answer = answer[-1].get("content", str(answer[-1]))

        print(f"\nShopMate: {answer}\n")

        messages.append({
            "role": "assistant",
            "content": str(answer)
        })

    except Exception as e:
        # Remove the failed user message from the history
        messages.pop()
        print(f"Error generating response: {e}")
