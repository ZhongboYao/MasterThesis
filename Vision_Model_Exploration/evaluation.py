from typing import Sequence
import base64
from openai import OpenAI
import api    
import pandas as pd
import util
import glob
import re
from PIL import Image  
import csv
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt 
import math
from fastembed import SparseTextEmbedding
from typing import List
from collections import Counter, defaultdict
from typing import Sequence, List, Dict

def exact_match(X: Sequence, y: Sequence) -> float:
    """
    Calculate accuracy by checking whether each corresponding element exactly matches.
    """
    correct = 0
    for truth, pred in zip(X, y):
        if truth == pred:
            correct += 1
    return correct / len(y)

def sparse_embed(txt: str) -> dict:
    """
    Generate the given txt's sparse embedding
    """
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    sparse_embedding = list(model.embed(txt))[0]
    return sparse_embedding

def sparse_dot_product(values1: Sequence, indices1: int, values2: Sequence, indices2: int) -> float:
    """
    Calculate the dot production of two vectors.
    """
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

def sparse_norm(values: Sequence) -> float:
    """
    Calculaate the L2 norm.
    """
    return math.sqrt(sum(val * val for val in values))

def sparse_cosine_similarity(embed1: dict, embed2: dict) -> float:
    """
    Calculate the cos similarity of two sparse embeddings.
    """
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

def compute_feature_accuracies(doc_anns: Sequence[Sequence[str]], gpt_anns: Sequence[Sequence[str]], features: List[str],
        choices_per_feature_freq: Dict[str, Dict[str, int]], threshold: float = 0.8, 
        average: str = "weighted", ignore_labels: List[str] = ["unselected"]
    ) -> Dict[str, float]:
    """
    For each feature (question), compute per-class recall:
      recall_c = TP_c / (TP_c + FN_c)
    Then average across classes either unweighted ("macro") or weighted by the dataset frequency of each class ("weighted").

    Parameters
    -----------
    doc_anns: List[list], #Cases * #Questions
        Ground truth labels.
    gpt_anns: List[list], #Cases * #Questions
        Predicted labels.
    features: list, #Questions
        List of feature names.
    choices_per_feature_freq: Dict[str(Question), Dict[str(Choice), int(frequency)]]
        Possible choices for each question with the corresponding frequency.
    threshold: float
        Cosine-similarity cutoff for matching labels
    average: str
        "macro" or "weighted" method.
    ignore_labels: List[str]
        A list of labels that shouldn't contribute to recall.

    Returns
    --------
    feature_recalls: Dict[str(Question), float(recall)]
        Mapping feature→averaged recall
    """
    # cache embeddings
    embedded: Dict[str, any] = {}

    def is_match(pred: str, true: str) -> bool:
        if pred not in embedded:
            embedded[pred] = sparse_embed(pred)
        if true not in embedded:
            embedded[true] = sparse_embed(true)
        return sparse_cosine_similarity(embedded[pred], embedded[true]) >= threshold

    # count true positives and total true examples per class
    tp_counts = {f: Counter() for f in features}
    true_counts = {f: Counter() for f in features}

    for true_row, pred_row in zip(doc_anns, gpt_anns):
        for idx, f in enumerate(features):
            t = true_row[idx]
            p = pred_row[idx]
            true_counts[f][t] += 1
            if is_match(p, t):
                tp_counts[f][t] += 1

    # compute recalls
    feature_recalls: Dict[str, float] = {}
    for f in features:
        recalls = []
        weights = []
        # iterate over all classes seen in the dataset
        for cls, freq in choices_per_feature_freq.get(f, {}).items():
            if freq == 0 or cls in ignore_labels:
                # skip zero-freq or explicitly ignored classes
                continue
            tp = tp_counts[f].get(cls, 0)
            recall_c = tp / true_counts[f].get(cls, freq)
            recalls.append(recall_c)
            weights.append(freq)
        if not recalls:
            feature_recalls[f] = 0.0
        elif average == "macro":
            feature_recalls[f] = sum(recalls) / len(recalls)
        else:  # weighted
            total_weight = sum(weights)
            feature_recalls[f] = sum(r * w for r, w in zip(recalls, weights)) / total_weight

    return feature_recalls

def compute_feature_acc_from_dict(acc_dict):
    feature_acc = {}
    for key, value in acc_dict.items():
        name = key
        total = value['total']
        acc = value['correct']/total
        feature_acc[name] = acc
    return feature_acc