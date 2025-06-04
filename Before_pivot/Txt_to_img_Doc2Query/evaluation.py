import vector_store as vs
import numpy as np
import math
import vector_store as vs
from tqdm import tqdm

def sparse_dot_product(values1, indices1, values2, indices2):
    i, j = 0, 0
    dot = 0.0
    while i < len(indices1) and j < len(indices2):
        if indices1[i] == indices2[j]:
            dot += values1[i] * values2[j]
            i += 1
            j += 1
        elif indices1[i] < indices2[j]:
            i += 1
        else:
            j += 1
    return dot

def sparse_norm(values):
    return math.sqrt(sum(val * val for val in values))

def sparse_cosine_similarity(embed1, embed2):
    v1 = embed1.values
    i1 = embed1.indices
    v2 = embed2.values
    i2 = embed2.indices
    dot = sparse_dot_product(v1, i1, v2, i2)
    norm1 = sparse_norm(v1)
    norm2 = sparse_norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  
    return dot / (norm1 * norm2)

embed_dict = {}
def embedding_list_hit(truth_list, prediction_list, sparse_sim_threshold):
    for truth in truth_list:
        if truth not in embed_dict:
            embed_dict[truth] = vs.sparse_embed(truth)
    
    for pred in prediction_list:
        if pred not in embed_dict:
            embed_dict[pred] = vs.sparse_embed(pred)
    
    truth_embs = [embed_dict[truth] for truth in truth_list]
    prediction_embs = [embed_dict[prediction] for prediction in prediction_list]
    sim_record = []
    for pred_emb in prediction_embs:
        for truth_emb in truth_embs:
            sparse_similarity = sparse_cosine_similarity(pred_emb, truth_emb)
            sim_record.append(sparse_similarity)
    if max(sim_record) > sparse_sim_threshold:
        return 1, max(sim_record)
    else:
        return 0, max(sim_record)

def plain_list_hit(truth_list, prediction_list):
    for entry in prediction_list:
        if entry in truth_list:
            return 1

    return 0
