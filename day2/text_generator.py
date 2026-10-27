from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    token=False,
    device=-1  
)
prompt = "Artificial Intelligence will"
result =  generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    pad_token_id=generator.tokenizer.eos_token_id 
)
print(results[0]["generated_text"])
