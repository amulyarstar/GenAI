from transformers import pipeline

qa_model = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0
)


messages = [
    {
        "role": "system",
        "content": "You are a helpful question-answering assistant. Answer clearly and accurately."
    },
    {
        "role": "user",
        "content": "What is Artificial Intelligence?"
    }
]


prompt = qa_model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


result = qa_model(
    prompt,
    max_new_tokens=100,
    do_sample=False
)


full_text = result[0]["generated_text"]

answer = (
    full_text.split("<|im_start|>assistant")[-1]
    .replace("<|im_end|>", "")
    .strip()
)

print("Question:", messages[-1]["content"])
print("Answer:", answer)
