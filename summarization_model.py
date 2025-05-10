#Dataset

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
val_data = dataset["validation"]

#Model & Tokenizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

#Preprocessing

def preprocess(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(preprocess, batched=True)
tokenized_val = val_data.map(preprocess, batched=True)


#Model Training

small_train_dataset = tokenized_train.select(range(10000))
small_val_dataset = tokenized_val.select(range(2000))


from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    save_total_limit=1,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

trainer.train()

#Evaluation

from rouge_score import rouge_scorer

ground_truth = "The quick brown fox jumps over the lazy dog."
predicted_text = "The fast brown fox leaps over the lazy dog."

# ------------------ ROUGE Score Calculation -------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
score = scorer.score(ground_truth, predicted_text)

rouge1_f1 = score['rouge1'].fmeasure
rouge2_f1 = score['rouge2'].fmeasure
rougeL_f1 = score['rougeL'].fmeasure

print(f"ROUGE-1 F1: {rouge1_f1:.4f}")
print(f"ROUGE-2 F1: {rouge2_f1:.4f}")
print(f"ROUGE-L F1: {rougeL_f1:.4f}")

# ------------------ Accuracy Calculation -------------------
accuracy = (predicted_text.strip() == ground_truth.strip())

print(f"Accuracy: {accuracy}")


#Results

from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

example_paragraph = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."

inputs = tokenizer(example_paragraph, max_length=512, truncation=True, return_tensors="pt")

inputs = {key: value.to(device) for key, value in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Predicted summary: {predicted_text}")



