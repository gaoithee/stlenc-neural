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
SAFE_THRESHOLD = 500 
OUTPUT_DIR = "./stlenc-training"

def main():
    wandb.init(
        project="STL-Encoder-Training",  
        name="transformer-stl-perfect-sync",
        job_type="train"
    )

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO_ID, config=config, trust_remote_code=True)

    full_dataset = load_dataset(DATASET_ID)
    raw_train = full_dataset["train"]
    raw_test = full_dataset["test"] if "test" in full_dataset else raw_train.train_test_split(test_size=0.1)["test"]

    def prepare_data(dataset, name=""):
        print(f"Filtraggio fisico {name}...")
        clean_list = []
        for i, ex in enumerate(dataset):
            token_out = tokenizer(ex["formula"], add_special_tokens=True, truncation=False)
            if len(token_out["input_ids"]) > SAFE_THRESHOLD:
                continue
            try:
                target = ex['embedding_1024']
                if isinstance(target, str):
                    target = ast.literal_eval(target.strip())
                if target is None or len(target) != 1024:
                    continue
                
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

    # --- TRAINER CON CALCOLO METRICHE ISTANTANEO ---
    class STLEncTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Inizializziamo accumulatori per le metriche di valutazione
            self.eval_mse_accumulator = []
            self.eval_cos_accumulator = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            
            if not isinstance(outputs, torch.Tensor):
                embeddings = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            else:
                embeddings = outputs
            
            if embeddings.dim() == 3:
                embeddings = embeddings[:, 0, :]
            
            loss = nn.functional.mse_loss(embeddings, labels)

            # Se siamo in modalità valutazione (model.training è False), 
            # calcoliamo le metriche istantaneamente Batch per Batch
            if not model.training:
                with torch.no_grad():
                    # Calcoliamo MSE e COS su questo specifico batch (già allineato!)
                    current_mse = loss.detach().item()
                    current_cos = nn.functional.cosine_similarity(embeddings, labels, dim=1).mean().item()
                    self.eval_mse_accumulator.append(current_mse)
                    self.eval_cos_accumulator.append(current_cos)

            return (loss, embeddings) if return_outputs else loss

    def compute_metrics(eval_pred):
        # Invece di usare eval_pred (che è disallineato), 
        # prendiamo i valori che il Trainer ha accumulato batch per batch
        # Accediamo all'istanza del trainer globale (un po' "hacky" ma risolutivo)
        mse_final = sum(trainer.eval_mse_accumulator) / max(len(trainer.eval_mse_accumulator), 1)
        cos_final = sum(trainer.eval_cos_accumulator) / max(len(trainer.eval_cos_accumulator), 1)
        
        # Resettiamo gli accumulatori per la prossima valutazione
        trainer.eval_mse_accumulator = []
        trainer.eval_cos_accumulator = []
        
        return {
            "mse_perfect_sync": mse_final,
            "cosine_similarity_sync": cos_final
        }

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

    print(f"Dataset pronti. Train: {len(train_processed)} | Eval: {len(test_processed)}")
    trainer.train()
    trainer.save_model("./final-model")
    wandb.finish()

if __name__ == "__main__":
    main()
