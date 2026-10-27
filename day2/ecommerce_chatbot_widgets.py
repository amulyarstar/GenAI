import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd
from transformers import pipeline

# 1. Hugging Face Transformers Pipeline
# If you switched to Gemma-3-1b, you can change model to "google/gemma-3-1b-it"
chatbot = pipeline(
    task="text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    device="cpu"
)

# 2. Load Dataset
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "FloZewi/E-commerce-Data-Analysis/master/"
    "ecommerce_product_dataset.csv"
)
df = pd.read_csv(DATA_URL)
df.columns = df.columns.str.strip()

# 3. System Prompt & Conversation State
SYSTEM_PROMPT = (
    "You are ShopMate, a friendly, helpful, and knowledgeable e-commerce shopping assistant. "
    "Answer questions about products, prices, ratings, discounts, stock availability, and categories. "
    "Use the product dataset provided in the conversation to answer factual questions. "
    "Never invent product names, prices, ratings, discounts, or stock information. "
    "If the required information is not provided, state that you cannot verify it from the dataset. "
    "Keep answers concise, friendly, and structured."
)

chat_history_state = [{"role": "system", "content": SYSTEM_PROMPT}]

# 4. In-Notebook UI Elements
chat_box = widgets.Output(layout=widgets.Layout(
    height="360px", 
    overflow_y="auto", 
    border="1px solid #3f3f46", 
    padding="12px",
    background_color="#18181b",
    border_radius="8px"
))

user_input = widgets.Text(
    placeholder="Ask ShopMate (e.g., 'Headphones', 'Smartwatches under $50')...",
    layout=widgets.Layout(width="80%")
)

send_button = widgets.Button(
    description="Send",
    button_style="primary",
    icon="paper-plane",
    layout=widgets.Layout(width="18%")
)

def render_message(sender, text, is_user=False):
    bg_color = "#27272a" if is_user else "#09090b"
    text_color = "#38bdf8" if is_user else "#a3e635"
    border = "#3b82f6" if is_user else "#22c55e"
    
    html = f"""
    <div style="margin-bottom: 12px; padding: 10px; background-color: {bg_color}; border-left: 4px solid {border}; border-radius: 6px; font-family: sans-serif; color: #f4f4f5;">
        <b style="color: {text_color};">{sender}:</b><br/>
        <div style="margin-top: 4px; white-space: pre-wrap; font-size: 14px; line-height: 1.4;">{text}</div>
    </div>
    """
    with chat_box:
        display(HTML(html))

# 5. Pipeline Query Handler
def on_send_clicked(b):
    query = user_input.value.strip()
    if not query:
        return
    
    user_input.value = ""
    render_message("You", query, is_user=True)
    
    # Token-based dataset matching (matches "headphones" even if user says "headphones under $20")
    tokens = [t.lower() for t in query.split() if len(t) > 2 and not t.startswith("$")]
    if tokens:
        mask = df.astype(str).apply(
            lambda col: col.str.lower().apply(lambda cell: any(tok in cell for tok in tokens))
        ).any(axis=1)
        matches = df[mask].head(6)
    else:
        matches = df.head(6)
    
    context = matches.to_string(index=False) if not matches.empty else "No matching products found in dataset."
    user_prompt = f"Product Dataset:\n{context}\n\nUser Question: {query}"
    
    chat_history_state.append({"role": "user", "content": user_prompt})
    
    # Transformers pipeline execution
    outputs = chatbot(
        chat_history_state,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.7
    )
    
    reply = outputs[0]["generated_text"][-1]["content"]
    chat_history_state.append({"role": "assistant", "content": reply})
    render_message("ShopMate 🛍️", reply, is_user=False)

send_button.on_click(on_send_clicked)

# Render UI
display(widgets.VBox([
    widgets.HTML("<h3 style='color:#38bdf8; margin: 4px 0;'>🛍️ ShopMate Assistant (Transformers Pipeline)</h3>"),
    chat_box, 
    widgets.HBox([user_input, send_button], layout=widgets.Layout(margin="8px 0 0 0"))
]))
