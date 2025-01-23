# Importing Libraries
import streamlit as st
from wikipediaapi import Wikipedia
import re 
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW


# Function to fetch data from Wikipedia
def fetch_wikipedia_content(topic, lang="en", user_agent='Your-User-Agent'):
    # Pass the user agent directly during initialization
    wiki_wiki = Wikipedia(user_agent=user_agent, language=lang)
    page = wiki_wiki.page(topic)
    if page.exists():
        return page.text
    else:
        return f"Page '{topic}' does not exist."

# Preprocess Text
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove references
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)  # Remove special characters
    return text.strip()

# Custom dataset for Wikipedia text
class WikiDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

    def __len__(self):
        return self.inputs["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.inputs["input_ids"][idx],
        }

# Function to train the model
def train_model(dataset, model, epochs=1, batch_size=4, lr=5e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].squeeze(1)
            attention_mask = batch["attention_mask"].squeeze(1)
            labels = batch["labels"].squeeze(1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        st.write(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Streamlit UI
st.title("WikiWeave - Wikipedia Data to Language Model Training")

# Step 1: Data Extraction
st.header("Step 1: Extract Data from Wikipedia")
topic = st.text_input("Enter a topic to fetch content:")
language = st.selectbox("Select language:", ["en", "es", "fr", "de"])

if st.button("Fetch Wikipedia Content"):
    with st.spinner("Fetching content..."):
        user_agent = "WikiWeave/1.0 (soodvedansh@gmail.com)"
        content = fetch_wikipedia_content(topic, language, user_agent)
        if content.startswith("Page"):
            st.error(content)
        else:
            st.success("Content fetched successfully!")
            processed_content = preprocess_text(content)
            st.subheader("Extracted Content:")
            st.write(content[:1000])
            st.subheader("Preprocessed Content:")
            st.write(processed_content[:1000])
            with open("wiki_data.txt", "w", encoding="utf-8") as file:
                file.write(processed_content)

            st.download_button(
                label="Download Preprocessed Content",
                data=processed_content,
                file_name=f"{topic}_processed.txt",
                mime="text/plain"
            )

# Step 2: Model Training
st.header("Step 2: Train Language Model")
if st.button("Train Model"):
    try:
        with st.spinner("Loading GPT-2 model and tokenizer..."):
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")

        with st.spinner("Preparing dataset..."):
            with open("wiki_data.txt", "r") as file:
                text = file.read()
            dataset = WikiDataset(text, tokenizer)

        with st.spinner("Training model..."):
            train_model(dataset, model, epochs=1)

        st.success("Model training completed!")
        st.download_button(
            label="Download Trained Model",
            data=torch.save(model.state_dict(), "trained_gpt2.pth"),
            file_name="trained_gpt2.pth",
            mime="application/octet-stream"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Step 3: Generate Predictions
st.header("Step 3: Generate Text with Trained Model")
prompt = st.text_input("Enter a prompt to generate text:")
if st.button("Generate Text"):
    try:
        with st.spinner("Loading model for text generation..."):
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.load_state_dict(torch.load("trained_gpt2.pth"))
            model.eval()
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("Generated Text:")
        st.write(generated_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")
