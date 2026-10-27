from transformers import pipeline
from huggingface_hub import login
from getpass import getpass

# Step 1: Login to Hugging Face
token = getpass("Enter your Hugging Face token: ")
login(token=token)
del token

print("Hugging Face login successful!")

# Step 2: Load the model
chatbot = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu"
)

print("Model loaded successfully!")

# Step 3: Define the cooking assistant
messages = [
    {
        "role": "system",
        "content": (
            "You are an expert, friendly, funny chef who knows about "
            "food and recipes. When the user gives you ingredients, "
            "suggest interesting recipes using those ingredients. "
            "Assume salt, sugar, and common spices are available. "
            "List ingredients, quantities, and step-by-step instructions. "
            "Keep answers clear and beginner-friendly."
        )
    }
]

print("\nWelcome to the new Cooking Assistant!")
print("Tell me your ingredients or ask for recipes.")
print("Type 'exit' to stop the conversation.\n")

# Step 4: Start the conversation
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("CookMate: Bye! Happy cooking!")
        break

    if not user_input:
        continue

    # Add the user's message
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate a response
    response = chatbot(
        messages,
        max_new_tokens=150,
        do_sample=False,
        return_full_text=False
    )

    answer = response[0]["generated_text"]

    print("\nCookMate:", answer, "\n")

    # Save the assistant's response for conversation history
    messages.append({
        "role": "assistant",
        "content": answer
    })
