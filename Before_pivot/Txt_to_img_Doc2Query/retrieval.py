class Retriever:
    def __init__(self, query, collection):
        self.query = query
        self.collection = collection
        self.found_docs = []
        self.reranked_docs = []
        self.filtered_embeddings = []
        self.filtered_contexts = []

    def similarity_search(self, k):
        self.found_docs = self.collection.similarity_search(self.query, k=k, with_vectors=True)
    

