import torch
import torch.nn as nn
import ast
import wandb
import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, TrainingArguments, Trainer

# --- CONFIGURAZIONE ---
REPO_ID = "saracandu/stlenc"
DATASET_ID = "saracandu/stl_formulae"
MAX_LENGTH = 512
SAFE_THRESHOLD = 500  # Soglia di sicurezza per i token
OUTPUT_DIR = "./stlenc-training"

def main():
    wandb.init(
        project="STL-Encoder-Training",  
        name="transformer-stl-safe-threshold",
        job_type="train"
    )

    print(f"Caricamento architettura da {REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO_ID, config=config, trust_remote_code=True)

    print(f"Caricamento dataset {DATASET_ID}...")
    full_dataset = load_dataset(DATASET_ID)
    
    raw_train = full_dataset["train"]
    raw_test = full_dataset["test"] if "test" in full_dataset else raw_train.train_test_split(test_size=0.1)["test"]

    # --- PREPROCESSING MANUALE PER ALLINEAMENTO TOTALE ---
    def prepare_data(dataset, name=""):
        print(f"Filtraggio fisico {name}...")
        clean_list = []
        for i, ex in enumerate(dataset):
            # Check lunghezza REALE (con token speciali)
            token_out = tokenizer(ex["formula"], add_special_tokens=True, truncation=False)
            
            if len(token_out["input_ids"]) > SAFE_THRESHOLD:
                continue

            try:
                target = ex['embedding_1024']
                if isinstance(target, str):
                    target = ast.literal_eval(target.strip())
                if target is None or len(target) != 1024:
                    continue
                
                # Padding manuale
                ids = token_out["input_ids"]
                mask = [1] * len(ids) + [0] * (MAX_LENGTH - len(ids))
                padded_ids = ids + [tokenizer.pad_token_id] * (MAX_LENGTH - len(ids))

                clean_list.append({
                    "input_ids": padded_ids,
                    "attention_mask": mask,
                    "labels": target
                })
            except:
                continue
        return Dataset.from_list(clean_list)

    train_processed = prepare_data(raw_train, "Train")
    test_processed = prepare_data(raw_test, "Eval")

    train_processed.set_format("torch")
    test_processed.set_format("torch")

    print(f"Dataset pronti. Train: {len(train_processed)} | Eval: {len(test_processed)}")

    class STLEncTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            
            # Pooling CLS
            if outputs.dim() == 3:
                outputs = outputs[:, 0, :]
            
            loss = nn.functional.mse_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # Sincronizzazione di emergenza (non dovrebbe servire con prepare_data)
        if logits.shape[0] != labels.shape[0]:
            ms = min(logits.shape[0], labels.shape[0])
            logits, labels = logits[:ms], labels[:ms]

        logits_t = torch.from_numpy(logits)
        labels_t = torch.from_numpy(labels)
        
        if logits_t.dim() == 3:
            logits_t = logits_t[:, 0, :]
            
        mse = nn.functional.mse_loss(logits_t, labels_t).item()
        # cos = nn.CosineSimilarity(dim=1).mean(torch.from_numpy(logits), torch.from_numpy(labels)).mean().item()
        # Nota: correggo l'uso della similarità per semplicità
        cos_sim = nn.functional.cosine_similarity(logits_t, labels_t).mean().item()
        
        return {"mse": mse, "avg_cosine_similarity": cos_sim}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="wandb",                
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        label_names=["labels"],
        remove_unused_columns=False
    )

    trainer = STLEncTrainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=test_processed,
        compute_metrics=compute_metrics,
    )

    print("Inizio addestramento...")
    trainer.train()
    trainer.save_model("./final-model")
    wandb.finish()

if __name__ == "__main__":
    main()
