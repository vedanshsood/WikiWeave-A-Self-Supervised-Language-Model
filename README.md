# WikiWeave

## Overview
WikiWeave is a comprehensive project that:
1. Extracts textual data from Wikipedia articles.
2. Preprocesses the extracted content to prepare it for model training.
3. Trains a self-supervised language model (based on GPT-2) to predict the next word in a sentence.
4. Provides a user-friendly UI built with Streamlit for seamless interaction.

This project is ideal for researchers, data scientists, and developers interested in natural language processing (NLP), language model training, and AI applications.

---

## Features
- **Wikipedia Data Extraction**: Fetch text from Wikipedia articles in multiple languages.
- **Data Preprocessing**: Clean and prepare text for training.
- **Model Training**: Fine-tune a GPT-2 language model on the processed text.
- **Text Generation**: Generate contextually relevant text using the trained model.
- **Streamlit UI**: Intuitive interface for all operations.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vedanshsood/WikiWeave.git
   cd WikiWeave
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## Requirements
- Python 3.8+
- Libraries:
  - `streamlit`
  - `wikipedia-api`
  - `torch`
  - `transformers`

---

## Usage

1. **Extract Data**: Enter a topic name and language to fetch Wikipedia content.
2. **Preprocess Data**: Automatically clean and prepare the extracted content.
3. **Train Model**: Fine-tune a GPT-2 language model using the preprocessed data.
4. **Generate Text**: Input a prompt and let the trained model generate predictions.

---

## File Structure
- `app.py`: Main Streamlit application.
- `requirements.txt`: List of dependencies.
- `wiki_data.txt`: Preprocessed Wikipedia data (created during execution).
- `trained_gpt2.pth`: Trained GPT-2 model weights (generated after training).

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Commit your changes.
4. Open a pull request.

---
