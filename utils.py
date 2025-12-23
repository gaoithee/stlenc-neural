import ast
import torch
from transformers import AutoTokenizer

def preprocess_dataset(example, tokenizer):
    # 1. Calcolo token e Filtraggio immediato
    ids = tokenizer.encode(example["formula"])
    max_len = 512
    if len(ids) > max_len:
        return {"input_ids": None} # Segnale per il filtro

    # 2. Parsing Target
    try:
        if isinstance(example['embedding_1024'], str):
            emb_list = ast.literal_eval(example['embedding_1024'].strip())
        else:
            emb_list = example['embedding_1024']
    except: return {"input_ids": None} # Scarta se l'embedding è corrotto

    # 3. Padding e Attention Mask
    input_ids = torch.tensor(ids)
    curr_len = input_ids.shape[0]
    pad_size = max_len - curr_len
    
    input_ids = torch.cat([input_ids, torch.full((pad_size,), 0, dtype=torch.long)])
    attention_mask = torch.cat([torch.ones(curr_len), torch.zeros(pad_size)]).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(emb_list, dtype=torch.float32)
    }