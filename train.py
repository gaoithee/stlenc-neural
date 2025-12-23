import torch
import ast
import os
import shutil
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from modeling_stlenc import STLEncoderModel
from configuration_stlenc import STLEncoderConfig
from tokenizer_stlenc import STLTokenizer

def preprocess_dataset(example, tokenizer):
    ids = tokenizer.encode(example["formula"])
    if len(ids) > 512:
        return {"input_ids": None, "attention_mask": None, "labels": None}

    try:
        emb = ast.literal_eval(example['embedding_1024'].strip()) if isinstance(example['embedding_1024'], str) else example['embedding_1024']
    except: return {"input_ids": None, "attention_mask": None, "labels": None}

    input_ids = torch.tensor(ids)
    pad_size = 512 - len(ids)
    input_ids = torch.cat([input_ids, torch.zeros(pad_size, dtype=torch.long)])
    attention_mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_size)]).long()

    return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(), "labels": emb}

def main():
    tokenizer = STLTokenizer()
    dataset = load_dataset("saracandu/stl_formulae", split="train")

    # Preprocessing e Filtraggio
    temp_ds = dataset.map(lambda x: preprocess_dataset(x, tokenizer), remove_columns=dataset.column_names)
    processed_ds = temp_ds.filter(lambda x: x["input_ids"] is not None)
    processed_ds.set_format(type='torch')
    
    print(f"Dataset pronto: {len(processed_ds)} esempi.")

    # Configurazione Modello
    config = STLEncoderConfig(vocab_size=tokenizer.vocab_size)
    model = STLEncoderModel(config)
    
    # Preparazione Ossatura per HF
    output_dir = "stl-ossatura"
    model.config.auto_map = {
        "AutoConfig": "configuration_stlenc.STLEncoderConfig",
        "AutoModel": "modeling_stlenc.STLEncoderModel"
    }
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    for f in ["modeling_stlenc.py", "configuration_stlenc.py", "tokenizer_stlenc.py"]:
        shutil.copy(f, os.path.join(output_dir, f))

    print(f"Ossatura salvata in {output_dir}. Ora puoi caricarla su HF.")

if __name__ == "__main__":
    main()
