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

class Evaluator:
    def __init__(self, truths, predictions):
        self.truths = truths
        self.predictions = predictions
        self.record = []
        self.score = []
        self.sparse_embeddings_dict = {}

    def list_acc(self, sparse_sim_threshold):
        pos = 0
        total = 0

        for truth, prediction_list in tqdm(zip(self.truths, self.predictions)):
            if truth not in self.sparse_embeddings_dict:
                truth_sparse_embedding = vs.sparse_embed(truth)
                self.sparse_embeddings_dict[truth] = truth_sparse_embedding

            for prediction in prediction_list:
                if prediction not in self.sparse_embeddings_dict:
                    prediction_sparse_embedding = vs.sparse_embed(prediction)
                    self.sparse_embeddings_dict[prediction] = prediction_sparse_embedding

            similarities_record = []
            prediction_list_embeddings = [self.sparse_embeddings_dict[prediction] for prediction in prediction_list]
            for prediction_embedding in prediction_list_embeddings:
                sparse_similarity = sparse_cosine_similarity(self.sparse_embeddings_dict[truth], prediction_embedding)
                similarities_record.append(sparse_similarity)
                max_sparse_similarity = max(similarities_record)

            if max_sparse_similarity > sparse_sim_threshold:
                pos += 1
                self.record.append(
                    {
                        "pred": prediction_list,
                        "truth": truth,
                        "sparse_similarity": max_sparse_similarity,
                        "correct": 1
                    }
                )
            else:
                self.record.append(
                    {
                        "pred": prediction_list,
                        "truth": truth,
                        "sparse_similarity": max_sparse_similarity,
                        "correct": 0
                    }
                )
                
            total += 1
        self.score.append({'list_acc': pos/total})
        return pos / total
    
    def single_acc(self, sparse_sim_threshold):
        pos = 0
        total = 0

        for truth, prediction in tqdm(zip(self.truths, self.predictions)):
            truth_sparse_embedding = vs.sparse_embed(truth)
            prediction_sparse_embedding = vs.sparse_embed(prediction)
            sparse_similarity = sparse_cosine_similarity(truth_sparse_embedding, prediction_sparse_embedding)
            if sparse_similarity > sparse_sim_threshold:
                pos += 1
                self.record.append(
                    {
                        "pred": prediction,
                        "truth": truth,
                        "sparse_similarity": sparse_similarity,
                        "correct": 1
                    }
                )
            else:
                self.record.append(
                    {
                        "pred": prediction,
                        "truth": truth,
                        "sparse_similarity": sparse_similarity,
                        "correct": 0
                    }
                )
                
            total += 1
        self.score.append({'list_acc': pos/total})
        return pos / total
            