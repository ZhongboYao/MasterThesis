import uuid
import Vision_Model_Exploration.api as api
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode
import math
import numpy as np
from openai import OpenAI
from fastembed import SparseTextEmbedding


openai_client = OpenAI(api_key=api.OPENAI_KEY)
qdrant_client = QdrantClient(url=api.QDRANT_URL, api_key=api.QDRANT_API)

def dense_embed(txt):
    response = openai_client.embeddings.create(
        input=txt,
        model="text-embedding-3-small"
    )
    dense_embedding = response.data[0].embedding
    return dense_embedding

def sparse_embed(txt):
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    sparse_embedding = list(model.embed(txt))[0]
    return sparse_embedding

def create_collection(collection_name, dense_embedding_dim):
    if qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted old version collection {collection_name}")

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dense_embedding_dim,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    print(f"Collection {collection_name} initialized.")


def add_txt(collection_name, chunk):
    dense_embedding = dense_embed(chunk.content)
    sparse_embedding = sparse_embed(chunk.content)

    # dense
    doc_id = f"{uuid.uuid4()}" 
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=doc_id, payload=chunk.__dict__, vector=dense_embedding)]
    )

    # sparse
    doc_id = f"{uuid.uuid4()}"
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=doc_id,
                payload=chunk.__dict__,
                vector={
                    "sparse": {  
                        "indices": list(sparse_embedding.indices),
                        "values": list(sparse_embedding.values)
                    }
                },
            ),
        ]
    )


def add_image(collection_name, image):
    caption_dense_embedding = dense_embed(image.caption)
    caption_sparse_embedding = sparse_embed(image.caption)

    # dense
    doc_id = f"{uuid.uuid4()}" 
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=doc_id, payload=image.__dict__, vector=caption_dense_embedding),]
    )

    # sparse
    doc_id = f"{uuid.uuid4()}"
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=doc_id,
                payload=image.__dict__,
                vector={
                    "sparse": {  
                        "indices": list(caption_sparse_embedding.indices),
                        "values": list(caption_sparse_embedding.values)
                    }
                },
            ),
        ]
    )

def get_collection(collection_name, dense_embedding_function, sparse_embedding_function, retrieval_mode=RetrievalMode.HYBRID):
    collection = QdrantVectorStore.from_existing_collection(
        embedding=dense_embedding_function,
        sparse_embedding=sparse_embedding_function,
        collection_name=collection_name,
        url=api.QDRANT_URL,
        api_key=api.QDRANT_API,
        retrieval_mode=retrieval_mode,
        sparse_vector_name="sparse"
    )
    return collection

def retrieve_payload(document, collection):
    qdrant_client = QdrantClient(url=api.QDRANT_URL, api_key=api.QDRANT_API)
    point_id = document.metadata["_id"]
    point = qdrant_client.retrieve(
        collection_name=collection.collection_name,
        ids=[point_id],
        with_payload=True,  
        with_vectors=False  
    )
    payload = point[0].payload
    return payload






