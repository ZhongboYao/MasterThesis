import vector_store as vs
import numpy as np
import math

def dense_cosine_similarity(embed1, embed2):
    dot_product = np.dot(embed1, embed2)
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    similarity =  dot_product / (norm1 * norm2)
    return similarity

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

class Evaluator:
    def __init__(self, truth, dense_threshold, sparse_threshold):
        self.truth = truth
        self.dense_threshold = dense_threshold
        self.sparse_threshold = sparse_threshold
        self.symptom_labels = []
        self.queries = []
        self.image_labels = []
        self.metrics = {}
        self.record = {}

        for item in truth:
            self.symptom_labels.append(item[0])
            self.queries.append(item[1])
            self.image_labels.append(item[2])

    def accuracy(self, preds, truths, record_name): 
        total = 0
        pos = 0
        self.record[record_name] = []

        for pred, truth in zip(preds, truths):
            if pred.split('/')[-1] == truth.split('/')[-1]:
                pos += 1
                self.record[record_name].append(
                {
                    "pred": pred,
                    "truth": truth,
                    "correct": 1
                }
            )
            else:
                self.record[record_name].append(
                {
                    "pred": pred,
                    "truth": truth,
                    "correct": 0
                }
            )
            total += 1
        return pos/total
    
    def semantic_acc(self, preds, truths, record_name):
        total = 0
        pos = 0
        self.record[record_name] = []

        print("Embedding predictions.")
        pred_dense_embeddings = []
        pred_sparse_embeddings = []
        for pred in preds:
            pred_dense_embeddings.append(vs.dense_embed(pred))
            pred_sparse_embeddings.append(vs.sparse_embed(pred))

        print("Embedding truths.")
        truth_dense_embeddings = []
        truth_sparse_embeddings = []
        for truth in truths:
            truth_dense_embeddings.append(vs.dense_embed(truth))
            truth_sparse_embeddings.append(vs.sparse_embed(truth))

        for pred_dense, truth_dense, pred_sparse, truth_sparse, pred_txt, truth_txt in zip(pred_dense_embeddings, truth_dense_embeddings, pred_sparse_embeddings, truth_sparse_embeddings, preds, truths):
            dense_similarity = dense_cosine_similarity(pred_dense, truth_dense)
            sparse_similarity = sparse_cosine_similarity(pred_sparse, truth_sparse)
            if dense_similarity >= self.dense_threshold or sparse_similarity >= self.sparse_threshold:
                pos += 1
                self.record[record_name].append(
                {
                    "pred": pred_txt,
                    "truth": truth_txt,
                    "dense_similarity": dense_similarity,
                    "sparse_similarity": sparse_similarity,
                    "correct": 1
                }
            )
            else:
                self.record[record_name].append(
                {
                     "pred": pred_txt,
                    "truth": truth_txt,
                    "dense_similarity": dense_similarity,
                    "sparse_similarity": sparse_similarity,
                    "correct": 0
                }
            )
            total += 1
        return pos/total

    def calculate_metrics(self, symptom_preds, image_preds):
        symptom_acc = self.accuracy(symptom_preds, self.symptom_labels, "symptom_acc")  
        image_acc = self.accuracy(image_preds, self.image_labels, "image_acc")
        symptom_semantic_acc = self.semantic_acc(symptom_preds, self.symptom_labels, "symptom_semantic_acc") 

        self.metrics["symptom_acc"] = symptom_acc
        self.metrics["image_acc"] = image_acc
        self.metrics["symptom_semantic_acc"] = symptom_semantic_acc
    
