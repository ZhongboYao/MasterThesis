import numpy as np
from sentence_transformers import CrossEncoder
from vector_store import retrieve_payload

class Retriever:
    def __init__(self, query, collection):
        self.query = query
        self.collection = collection
        self.found_docs = []
        self.reranked_docs = []

    def similarity_search(self, k):
        self.found_docs = self.collection.similarity_search(self.query, k=k, with_vectors=True)
    
    def rerank(self, content_key, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        cross_encoder = CrossEncoder(model_name)
        rerank_input = []
        for document in self.found_docs:
            payload = retrieve_payload(document, self.collection)
            context = payload.get(content_key, "")
            rerank_input.append((self.query, context))
        scores = cross_encoder.predict(rerank_input)

        for i, doc in enumerate(self.found_docs):
            doc.metadata["cross_score"] = scores[i]

        self.reranked_docs = sorted(
            self.found_docs,
            key=lambda x: x.metadata["cross_score"],
            reverse=True
        )

    def cos_filtering(self, embedding_fn, content_key, threshold, k):
        self.filtered_embeddings = []
        self.filtered_contexts = []

        for doc in self.reranked_docs:
            payload = retrieve_payload(doc, self.collection)
            context = payload.get(content_key, "")
            embedding = embedding_fn(context)  

            if cosine_similarity_filter(embedding, self.filtered_embeddings, threshold):
                self.filtered_embeddings.append(embedding)
                self.filtered_contexts.append(context)

            if len(self.filtered_contexts) >= k:
                break
    
def cosine_similarity_filter(candidate, selected_vectors, threshold):
    if not selected_vectors :
        return True
    
    for vector in selected_vectors:
        dot_product = np.dot(candidate, vector)
        norm_candidate = np.linalg.norm(candidate)
        norm_vector = np.linalg.norm(vector)
        similarity =  dot_product / (norm_candidate * norm_vector)
        if similarity > threshold:
            return False
    return True

