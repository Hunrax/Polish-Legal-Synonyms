from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

model_names = [
    "allegro/herbert-large-cased",
    "dkleczek/bert-base-polish-cased-v1",
    "nlpaueb/legal-bert-base-uncased"
]

def get_embeddings(words: set[str], model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")

    word_embeddings = {}
    for word in words:
        tokens = tokenizer(word, return_tensors='pt', add_special_tokens=False).to("cuda")
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=0)
            word_embeddings[word] = normalized_embeddings.cpu().numpy()

    return word_embeddings


def get_contextual_embeddings(texts: list[str], target_words: set[str], model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda")

    collected_vectors = defaultdict(list)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, 
                           truncation=True, max_length=512).to("cuda")
        offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        for word in target_words:
            start_idx = 0
            while True:
                start_char = text.find(word, start_idx)
                if start_char == -1: break
                
                end_char = start_char + len(word)
                token_indices = [i for i, (s, e) in enumerate(offsets) 
                                 if s >= start_char and e <= end_char and s != e]
                
                if token_indices:
                    vec = outputs.last_hidden_state[0, token_indices].mean(dim=0)
                    collected_vectors[word].append(vec)
                
                start_idx = end_char

    final_embeddings = {}
    for word, vecs in collected_vectors.items():
        if vecs:
            stacked_vecs = torch.stack(vecs)
            avg_vec = stacked_vecs.mean(dim=0)
            normalized_vec = F.normalize(avg_vec, p=2, dim=0)
            final_embeddings[word] = normalized_vec.cpu().numpy()
            
    return final_embeddings
