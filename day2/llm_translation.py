from transformers import pipeline

# Using a text-generation pipeline with an instruct model for translation
translator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=0
)

messages = [
    {"role": "system", "content": "You are a helpful translator. Translate the user's text from English to German."},
    {"role": "user", "content": "Hi!! I am amulya"}
]

prompt = translator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
result = translator(prompt, max_new_tokens=50, do_sample=False)

# Extract and print the generated translation
full_text = result[0]["generated_text"]
print(full_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip())
