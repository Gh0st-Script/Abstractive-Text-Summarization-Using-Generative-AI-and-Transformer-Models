# Abstractive-Text-Summarization-Using-Generative-AI-and-Transformer-Models

# Abstractive Text Summarization Using BART

This project implements an **abstractive text summarization** system using the [facebook/bart-base](https://huggingface.co/facebook/bart-base) transformer model, trained on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail). The model is fine-tuned using Hugging Face’s `Trainer` API and evaluated with ROUGE scores to determine summarization quality.

---

## Project Overview

- Trains a BART-based transformer model for text summarization.
- Fine-tunes using a subset of the CNN/DailyMail dataset.
- Evaluates generated summaries with ROUGE-1, ROUGE-2, and ROUGE-L.
- Includes an example inference for demo/testing.
- Tracks whether the predicted summary matches the reference exactly (for demonstration).

---

## Example Output

![image](https://github.com/user-attachments/assets/6011ff5b-2043-4d2a-aae9-6b3c5641f50c)


