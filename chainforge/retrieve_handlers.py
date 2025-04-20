# For Retrieval Methods
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from gensim.utils import simple_preprocess

class EmbeddingModelRegistry:
    _models = {}
    
    @classmethod
    def register(cls, model_name):
        def decorator(embedding_func):
            cls._models[model_name] = embedding_func
            return embedding_func
        return decorator
        
    @classmethod
    def get_embedder(cls, model_name):
        return cls._models.get(model_name)
    
    @classmethod
    def list_models(cls):
        return list(cls._models.keys())

@EmbeddingModelRegistry.register("huggingface")
def huggingface_embedder(texts, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generate embeddings using HuggingFace Transformers.
    
    Args:
        texts: List of text strings to embed
        model_name: HuggingFace model name/path (default: sentence-transformers/all-mpnet-base-v2)
        
    Returns:
        List of embeddings for each text
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        print(f"Using HuggingFace model: {model_name} for {len(texts)} texts")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for t in batch_texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, 
                                  max_length=512)  # Add max_length for safety
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use mean pooling by default
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                batch_embeddings.append(emb)
                
            embeddings.extend(batch_embeddings)
            
        return embeddings
    except Exception as e:
        print(f"HuggingFace embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate HuggingFace embeddings: {str(e)}")

@EmbeddingModelRegistry.register("OpenAI Embeddings")
def openai_embedder(texts, model_name="text-embedding-ada-002"):
    """
    Generate embeddings using OpenAI Embeddings.
    
    Args:
        texts: List of text strings to embed
        model_name: OpenAI embedding model to use (default: text-embedding-ada-002)
        
    Returns:
        List of embeddings for each text
    """
    try:
        import openai
        print(f"Using OpenAI model: {model_name} for {len(texts)} texts")
        
        embeddings = []
        # Process in batches of 16 to stay within rate limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for t in batch_texts:
                resp = openai.Embedding.create(input=t, model=model_name)
                emb = resp["data"][0]["embedding"]
                batch_embeddings.append(emb)
                
            embeddings.extend(batch_embeddings)
            
        return embeddings
    except Exception as e:
        print(f"OpenAI embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate OpenAI embeddings: {str(e)}")

@EmbeddingModelRegistry.register("Cohere Embeddings")
def cohere_embedder(texts, model_name="embed-english-v2.0"):
    """
    Generate embeddings using Cohere Embeddings.
    
    Args:
        texts: List of text strings to embed
        model_name: Cohere embedding model to use (default: embed-english-v2.0)
        
    Returns:
        List of embeddings for each text
    """
    try:
        import cohere
        print(f"Using Cohere model: {model_name} for {len(texts)} texts")
        
        # Get API key from environment or settings
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            from flask import current_app
            api_key = current_app.config.get("COHERE_API_KEY")
            
        if not api_key:
            raise ValueError("Cohere API key not found in environment or app config")
            
        co = cohere.Client(api_key)
        
        batch_size = 32  
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = co.embed(texts=batch_texts, model=model_name)
            embeddings.extend(response.embeddings)
            
        return embeddings
    except Exception as e:
        print(f"Cohere embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate Cohere embeddings: {str(e)}")

@EmbeddingModelRegistry.register("Sentence Transformers")
def sentence_transformers_embedder(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings using Sentence Transformers.
    
    Args:
        texts: List of text strings to embed
        model_name: Sentence Transformers model name (default: all-MiniLM-L6-v2)
        
    Returns:
        List of embeddings for each text
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Using SentenceTransformer model: {model_name} for {len(texts)} texts")
        
        model = SentenceTransformer(model_name)
        
        # Process in reasonable batch sizes
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = model.encode(batch_texts).tolist()
            embeddings.extend(embeddings)
            
        return embeddings
    except Exception as e:
        print(f"SentenceTransformer embedder failed: {str(e)}")
        raise ValueError(f"Failed to generate SentenceTransformer embeddings: {str(e)}")

# Define a registry for retrieval methods
class RetrievalMethodRegistry:
    _methods = {}
    
    @classmethod
    def register(cls, method_name):
        def decorator(handler_func):
            cls._methods[method_name] = handler_func
            return handler_func
        return decorator
        
    @classmethod
    def get_handler(cls, method_name):
        return cls._methods.get(method_name)

@RetrievalMethodRegistry.register("bm25")
def handle_bm25(chunk_objs, query_objs, settings):
    top_k = settings.get("top_k", 5)
    k1 = settings.get("bm25_k1", 1.5)
    b = settings.get("bm25_b", 0.75)
    # Extract text from objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]

    # Preprocess corpus once
    tokenized_corpus = [simple_preprocess(doc) for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    results = []
    for query_obj in query_objs:
        tokenized_query = simple_preprocess(query_obj.get("text", ""))
        raw_scores = bm25.get_scores(tokenized_query)
        # Normalize scores
        max_score = max(raw_scores) if raw_scores.any() and max(raw_scores) > 0 else 1
        normalized_scores = [score / max_score for score in raw_scores]
        
        # Build result objects with all the necessary metadata
        retrieved = []
        scored_chunks = sorted(zip(chunk_objs, normalized_scores), key=lambda x: x[1], reverse=True)
        
        for chunk, similarity in scored_chunks[:top_k]:
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(similarity),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("tfidf")
def handle_tfidf(chunk_objs, query_objs, settings):
    top_k = settings.get("top_k", 5)
    max_features = settings.get("max_features", 500)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    # Create and fit vectorizer once for all queries
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    results = []
    for query_obj in query_objs:
        query_vec = vectorizer.transform([query_obj.get("text", "")])
        sims = (tfidf_matrix * query_vec.T).toarray().flatten()
        
        # Normalize scores
        max_sim = sims.max() if sims.size > 0 and sims.max() > 0 else 1
        normalized_sims = sims / max_sim
        
        # Build result objects
        retrieved = []
        ranked_idx = normalized_sims.argsort()[::-1][:top_k]
        
        for i in ranked_idx:
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(normalized_sims[i]),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("boolean")
def handle_boolean(chunk_objs, query_objs, settings):
    top_k = settings.get("top_k", 5)
    required_match_count = settings.get("required_match_count", 1)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    results = []
    for query_obj in query_objs:
        q_tokens = set(simple_preprocess(query_obj.get("text", "")))
        if len(q_tokens) < required_match_count:
            # Not enough tokens in query to match the required count
            results.append({'query_object': query_obj, 'retrieved_chunks': []})
            continue
            
        scored = []
        for i, c in enumerate(chunk_texts):
            c_tokens = set(simple_preprocess(c))
            matches = len(q_tokens.intersection(c_tokens))
            if matches >= required_match_count:
                score = matches / (len(c_tokens) + 1e-9)  # Normalize by document length
                scored.append((i, score))
                
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        retrieved = []
        if scored:
            max_score = scored[0][1]
            for i, score in scored[:top_k]:
                chunk = chunk_objs[i]
                normalized_score = score / max_score if max_score > 0 else 0
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(normalized_score),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("overlap")
def handle_keyword_overlap(chunk_objs, query_objs, settings):
    top_k = settings.get("top_k", 5)
    
    # Extract text from chunk objects
    chunk_texts = [chunk.get("text", "") for chunk in chunk_objs]
    
    results = {}
    for query_obj in query_objs:
        q_tokens = set(simple_preprocess(query_obj.get("text", "")))
        scored = []
        
        for i, c in enumerate(chunk_texts):
            c_tokens = set(simple_preprocess(c))
            overlap = len(q_tokens.intersection(c_tokens))
            scored.append((i, overlap))
            
        # Sort by overlap count
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        retrieved = []
        if scored and scored[0][1] > 0:  # Ensure max score > 0
            max_score = scored[0][1]
            for i, score in scored[:top_k]:
                chunk = chunk_objs[i]
                normalized_score = score / max_score
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(normalized_score),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
        else:
            # No overlaps found
            retrieved = []
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
import math


# Helper functions for similarity calculations
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

def manhattan_distance(vec1, vec2):
    """Compute Manhattan distance between two vectors"""
    return sum(abs(a - b) for a, b in zip(vec1, vec2))

def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

@RetrievalMethodRegistry.register("cosine")
def handle_cosine_similarity(chunks, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using cosine similarity between embeddings.
    
    This implementation uses a min-heap to keep only the top-k results in memory.
    """
    top_k = settings.get("top_k", 5)
    results = []
    
    for (query_obj, query_emb) in zip(query_objs, query_embeddings):
        # Use a min heap to keep track of top k results
        min_heap = []
        
        # Calculate similarities and maintain heap of size top_k
        for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
            sim = cosine_similarity(chunk_emb, query_emb)
            
            # If heap is not full, add the item
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            # If similarity is higher than the smallest in heap, replace it
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results (highest similarity first)
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunks[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("manhattan")
def handle_manhattan(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using Manhattan distance between embeddings.
    """
    top_k = settings.get("top_k", 5)
    results = []
    
    for query_obj, query_emb in zip(query_objs, query_embeddings):
        # Use a min heap to keep track of top k results
        min_heap = []
        
        # Calculate similarities and maintain heap of size top_k
        for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
            # Lower Manhattan distance = higher similarity
            distance = manhattan_distance(chunk_emb, query_emb)
            sim = 1.0 / (1.0 + distance)  # Transform to similarity score
            
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("euclidean")
def handle_euclidean(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using Euclidean distance between embeddings.
    """
    top_k = settings.get("top_k", 5)
    results = []
    
    for query_obj, query_emb in zip(query_objs, query_embeddings):
        min_heap = []
        
        for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
            distance = euclidean_distance(chunk_emb, query_emb)
            sim = 1.0 / (1.0 + distance)  # Transform to similarity score
            
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (sim, i))
            elif sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i))
        
        # Convert heap to sorted results
        retrieved = []
        for sim, i in sorted(min_heap, reverse=True):
            chunk = chunk_objs[i]
            retrieved.append({
                "text": chunk.get("text", ""),
                "similarity": float(sim),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": chunk.get("chunkId", ""),
            })
        
        results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    
    return results

@RetrievalMethodRegistry.register("clustered")
def handle_clustered(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using a combination of query similarity and cluster similarity.
    """
    top_k = settings.get("top_k", 5)
    n_clusters = settings.get("n_clusters", 3)
    query_coeff = settings.get("query_coeff", 0.6)
    center_coeff = settings.get("center_coeff", 0.4)
    results = []
    
    # Convert embeddings to numpy array for clustering
    X = np.array(chunk_embeddings)
    
    # Only perform clustering if we have enough samples
    if len(X) >= 2:
        n_clusters = min(n_clusters, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
        
        for query_obj, query_emb in zip(query_objs, query_embeddings):
            min_heap = []
            query_emb_np = np.array(query_emb).reshape(1, -1)
            
            for i, (chunk, chunk_emb) in enumerate(zip(chunk_objs, chunk_embeddings)):
                # Calculate similarity to query
                chunk_emb_np = np.array(chunk_emb).reshape(1, -1)
                query_sim = float(sklearn_cosine(chunk_emb_np, query_emb_np)[0][0])
                
                # Calculate similarity to cluster center
                center_sim = float(sklearn_cosine(
                    chunk_emb_np, 
                    cluster_centers[labels[i]].reshape(1, -1)
                )[0][0])
                
                # Combined similarity score (weighted)
                combined_sim = query_coeff * query_sim + center_coeff * center_sim
                
                if len(min_heap) < top_k:
                    heapq.heappush(min_heap, (combined_sim, i))
                elif combined_sim > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (combined_sim, i))
            
            # Convert heap to sorted results
            retrieved = []
            for sim, i in sorted(min_heap, reverse=True):
                chunk = chunk_objs[i]
                retrieved.append({
                    "text": chunk.get("text", ""),
                    "similarity": float(sim),
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk.get("chunkId", ""),
                })
            
            results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})
    return results


# VECTOR STORE RETRIEVAL METHODS
from langchain_core.embeddings import Embeddings
from typing import List
import numpy as np

# --- Define DummyEmbeddings Class ---
class DummyEmbeddings(Embeddings):
    """
    A dummy embedding class implementing the LangChain Embeddings interface.
    Used when pre-computed embeddings are provided.
    Returns zero vectors of the specified dimension.
    """
    def __init__(self, dimension: int):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")
        self.dimension = dimension
        # Store a zero vector template for efficiency
        self._zero_vector = [0.0] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return zero vectors for a list of documents."""
        return [self._zero_vector for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return a single zero vector for a query."""
        return self._zero_vector

# FAISS
import faiss
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

@RetrievalMethodRegistry.register("faiss")
def handle_faiss(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using FAISS with L2 (Euclidean) or IP (Inner Product) metric.
    """
    top_k = settings.get("top_k", 5)
    user_requested_metric = settings.get("metric", "l2").lower()
    if user_requested_metric not in ["l2", "ip"]:
        print(f"Warning: Invalid FAISS metric '{user_requested_metric}' specified. Defaulting to 'l2'.")
        user_requested_metric = "l2"

    faiss_mode = settings.get("faissMode", "create").lower()
    faiss_path = settings.get("faissPath", "") # Path to the FOLDER
    # Ensure similarity threshold is a float between 0.0 and 1.0
    try:
        similarity_threshold = float(settings.get("similarity_threshold", 0)) / 100.0
        similarity_threshold = max(0.0, min(1.0, similarity_threshold))
    except ValueError:
        print("Warning: Invalid similarity_threshold value. Defaulting to 0.")
        similarity_threshold = 0.0

    # Consistent result structure initialization
    results = []

    # Basic Input Validation
    if not chunk_objs or not chunk_embeddings:
         print("Error: chunk_objs or chunk_embeddings are empty.")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
    if not query_objs or not query_embeddings:
         print("Error: query_objs or query_embeddings are empty.")
         # Return empty results for potentially valid chunks if queries are missing
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    try:
        if not isinstance(chunk_embeddings[0], list) or not isinstance(query_embeddings[0], list):
             raise TypeError("Embeddings should be lists of lists of floats.")
        dimension = len(chunk_embeddings[0])
        query_dimension = len(query_embeddings[0])
    except (IndexError, TypeError) as e:
         print(f"Error validating embedding structure: {e}")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    if dimension != query_dimension:
         print(f"Error: Embedding dimension mismatch: Chunks({dimension}), Queries({query_dimension})")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    chunk_embeddings_np = np.array(chunk_embeddings).astype('float32')
    query_embeddings_np = np.array(query_embeddings).astype('float32')

    try:
        dummy_embeddings = DummyEmbeddings(dimension=dimension)
    except ValueError as e:
         print(f"Error: Failed to initialize DummyEmbeddings: {e}")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    vector_store = None

    # === Step 1: Initialize LangChain FAISS Vector Store ===
    try:
        if faiss_mode == "load":
            index_file = os.path.join(faiss_path, "index.faiss")
            pkl_file = os.path.join(faiss_path, "index.pkl")
            if not faiss_path or not os.path.exists(index_file) or not os.path.exists(pkl_file):
                print(f"Error: FAISS index not found in folder '{faiss_path}' for loading.")
                return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

            vector_store = FAISS.load_local(
                folder_path=faiss_path,
                embeddings=dummy_embeddings,
                index_name="index",
                allow_dangerous_deserialization=True
            )

            # Check Loaded Index Dimension
            loaded_dimension = vector_store.index.d
            if loaded_dimension != dimension:
                 print(f"Error: Dimension mismatch: Loaded index({loaded_dimension}), Provided queries({dimension})")
                 return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

            # Check if loaded metric matches requested metric (optional, but good practice)
            loaded_metric_type = vector_store.index.metric_type
            loaded_metric_str = "l2" if loaded_metric_type == faiss.METRIC_L2 else "ip" if loaded_metric_type == faiss.METRIC_INNER_PRODUCT else "unknown"

            if loaded_metric_str != user_requested_metric and loaded_metric_str != "unknown":
                 print(f"Warning: Loaded FAISS index metric ('{loaded_metric_str}') does not match requested metric ('{user_requested_metric}'). Using the loaded index's metric for search.")
            elif loaded_metric_str == "unknown":
                 print(f"Warning: Loaded FAISS index has an unknown metric type ({loaded_metric_type}). Proceeding with caution, interpreting as L2.")
                 # Force interpretation? Or error out? Defaulting to L2 interpretation for now.

        elif faiss_mode == "create":
            texts = [chunk.get("text", "") for chunk in chunk_objs]
            metadatas = [{"docTitle": chunk.get("docTitle", ""), "chunkId": chunk.get("chunkId", str(i))} for i, chunk in enumerate(chunk_objs)]

            # Create index based on user_requested_metric
            if user_requested_metric == "ip":
                print("Creating FAISS index with IP metric (normalizing vectors for Cosine Similarity).")
                # IMPORTANT: Normalize vectors for IP index to compute cosine similarity
                faiss.normalize_L2(chunk_embeddings_np)
                index = faiss.IndexFlatIP(dimension)
            else: # Default or user_requested_metric == "l2"
                print("Creating FAISS index with L2 metric.")
                index = faiss.IndexFlatL2(dimension)

            docstore = InMemoryDocstore({str(i): Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))})
            index_to_docstore_id = {i: str(i) for i in range(len(texts))}

            index.add(chunk_embeddings_np) # Add potentially normalized embeddings

            vector_store = FAISS(
                embedding_function=dummy_embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )

            # Save the newly created index if a path is provided
            if faiss_path:
                try:
                    if not os.path.isdir(faiss_path):
                         os.makedirs(faiss_path, exist_ok=True)
                    vector_store.save_local(folder_path=faiss_path, index_name="index")
                    print(f"FAISS index saved to {faiss_path}")
                except Exception as e_save:
                     print(f"Warning: Error saving FAISS index to {faiss_path}: {e_save}. Retrieval will continue.")
                     # Continue even if saving fails

        else:
             print(f"Error: Invalid faissMode: '{faiss_mode}'. Must be 'create' or 'load'.")
             return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    except Exception as e_init:
        print(f"Error during FAISS index initialization ({faiss_mode} mode): {e_init}")
        return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]


    if not vector_store or not isinstance(vector_store.index, faiss.Index):
         print("Error: Failed to initialize a valid FAISS vector store object.")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    # === Step 2: Determine the metric of the active index for searching ===
    search_metric_type = vector_store.index.metric_type
    if search_metric_type == faiss.METRIC_L2:
        search_metric = "l2"
    elif search_metric_type == faiss.METRIC_INNER_PRODUCT:
        search_metric = "ip"
    else:
        print(f"Warning: Active FAISS index has unexpected metric type {search_metric_type}. Defaulting to L2 interpretation for search.")
        search_metric = "l2" # Fallback interpretation

    print(f"Performing FAISS search using {search_metric.upper()} metric.")

    # === Step 3: Perform FAISS Retrieval ===
    for query_obj, q_embedding in zip(query_objs, query_embeddings_np):
        retrieved = []
        try:
            query_vec = q_embedding.reshape(1, -1).astype('float32')

            # IMPORTANT: Normalize query vector if using IP index for cosine similarity
            if search_metric == "ip":
                faiss.normalize_L2(query_vec)

            # Use LangChain's search method which handles index interaction
            # Note: LangChain's similarity_search_with_score_by_vector returns:
            # - For L2 index: Lower score means *more* similar (distance)
            # - For IP index (with normalized vectors): Higher score means *more* similar (cosine similarity)
            search_results = vector_store.similarity_search_with_score_by_vector(
                embedding=query_vec[0], # Pass the 1D vector
                k=top_k
            )

            # === Step 4: Convert results, interpret score based on metric, apply threshold ===
            for doc, score in search_results:
                similarity_score = 0.0
                raw_score = float(score)

                if search_metric == "l2":
                    # Convert L2 distance to similarity score (common method: 1 / (1 + distance))
                    # Ensure distance is non-negative
                    l2_distance = max(0.0, raw_score)
                    similarity_score = 1.0 / (1.0 + l2_distance)
                elif search_metric == "ip":
                    # Score from IP index (after normalization) is cosine similarity
                    # Clamp to [0, 1] range as cosine similarity should be within [-1, 1]
                    # but embeddings models often produce values in [0, 1] or normalization ensures this.
                    # Clamping defensively.
                    similarity_score = max(0.0, min(1.0, raw_score))

                # Apply the user-defined similarity threshold
                if similarity_score >= similarity_threshold:
                    retrieved.append({
                        "text": doc.page_content,
                        "similarity": round(similarity_score, 6), # Standardized rounding
                        "docTitle": doc.metadata.get("docTitle", ""),
                        "chunkId": doc.metadata.get("chunkId", ""),
                    })

            # Sort final results by similarity AFTER score conversion and thresholding
            retrieved.sort(key=lambda x: x["similarity"], reverse=True)
            results.append({'query_object': query_obj, 'retrieved_chunks': retrieved})

        except Exception as e_search:
            query_text_preview = query_obj.get("text", "N/A")[:70] + "..." if isinstance(query_obj, dict) else str(query_obj)[:70] + "..."
            print(f"Error during similarity search for query '{query_text_preview}': {e_search}")
            results.append({'query_object': query_obj, 'retrieved_chunks': []}) # Append empty result for this query

    return results

# PINECONE
@RetrievalMethodRegistry.register("pinecone")
def handle_pinecone(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using Pinecone with adaptable behavior based on mode: create, load.
    Includes smarter waiting based on index stats polling.
    Aligned with standard handler signature and return structure.
    """
    print("[DEBUG] Entered handle_pinecone function.")

    # 1. Extract settings
    top_k = settings.get("top_k", 5)
    # Pinecone metric used for index creation and score interpretation
    similarity_function = settings.get("pineconeSimilarity", "cosine").lower()
    # Threshold: Assume user provides 0-100, convert later based on metric
    raw_similarity_threshold = settings.get("similarity_threshold", 0)
    try:
        # Validate it's a number first
        raw_similarity_threshold = float(raw_similarity_threshold)
    except ValueError:
        print(f"[WARN] Invalid similarity_threshold value '{raw_similarity_threshold}'. Defaulting to 0.")
        raw_similarity_threshold = 0.0

    pinecone_api_key = settings.get("pineconeApiKey", "")
    pinecone_env = settings.get("pineconeEnvironment", "us-east-1") # Note: env is often deprecated for API key based routing
    pinecone_index_name = settings.get("pineconeIndex", "")
    pinecone_namespace = settings.get("pineconeNamespace", "")  # optional
    pinecone_mode = settings.get("pineconeMode", "create").lower()  # "create", "load"

    # --- Smarter Wait Settings ---
    polling_interval_seconds = settings.get("pineconePollingInterval", 3) # Check every 3 seconds
    max_wait_time_seconds = settings.get("pineconeMaxWaitTime", 120) # Max wait 2 minutes

    print(f"[DEBUG] Pinecone settings extracted:")
    print(f"  top_k = {top_k}")
    print(f"  raw_similarity_threshold = {raw_similarity_threshold}")
    print(f"  pinecone_api_key = {'(HIDDEN)' if pinecone_api_key else '(MISSING)'}")
    # print(f"  pinecone_env = {pinecone_env}") # Environment often less relevant now
    print(f"  pinecone_index_name = {pinecone_index_name}")
    print(f"  pinecone_namespace = {pinecone_namespace if pinecone_namespace else '(Default)'}")
    print(f"  similarity_function = {similarity_function}")
    print(f"  pinecone_mode = {pinecone_mode}")
    print(f"  polling_interval = {polling_interval_seconds}s, max_wait_time = {max_wait_time_seconds}s")

    # Consistent result structure initialization
    final_results = []

    # Basic Input Validation
    if not pinecone_api_key or not pinecone_index_name:
        print("[ERROR] Missing Pinecone API key or index name. Aborting.")
        return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs] # Consistent error return
    if not chunk_objs or not chunk_embeddings: # Check both, although upsert might be skipped if mode is load
         print("[WARN] chunk_objs or chunk_embeddings list is empty. Upsert may be skipped if creating.")
         # Allow proceeding in 'load' mode even if chunks are empty, but error if creating?
         if pinecone_mode == "create" and (not chunk_objs or not chunk_embeddings):
              print("[ERROR] Cannot create index with empty chunks/embeddings. Aborting.")
              return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
    if not query_objs or not query_embeddings:
         print("[ERROR] query_objs or query_embeddings are empty. Cannot perform retrieval. Aborting.")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs] # Consistent error return

    try:
        dimension = len(chunk_embeddings[0]) if chunk_embeddings else None # Get dimension if possible
        query_dimension = len(query_embeddings[0])

        if dimension is not None and dimension != query_dimension:
             print(f"[ERROR] Embedding dimension mismatch: Chunks({dimension}), Queries({query_dimension}). Aborting.")
             return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
        elif dimension is None and pinecone_mode == 'create':
             print(f"[ERROR] Cannot determine embedding dimension from empty chunk_embeddings in create mode. Aborting.")
             return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
        elif dimension is None:
             dimension = query_dimension # Use query dimension if chunks are empty in load mode

    except (IndexError, TypeError) as e:
         print(f"[ERROR] Error validating embedding structure or getting dimension: {e}. Aborting.")
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]


    # 2. Initialize Pinecone
    print("[DEBUG] Initializing Pinecone client...")
    try:
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Pinecone client: {e}. Check API key and connectivity. Aborting.")
        return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    # 3. Check/Create Index
    index = None
    upsert_chunks_flag = False # Flag to control upsert logic

    try:
        print("[DEBUG] Checking existing Pinecone indexes...")
        existing_indexes_info = pc.list_indexes()
        existing_index_names = [idx["name"] for idx in existing_indexes_info]
        index_exists = pinecone_index_name in existing_index_names

        if pinecone_mode == "create":
            if index_exists:
                print(f"[DEBUG] Deleting existing Pinecone index '{pinecone_index_name}'...")
                try:
                    pc.delete_index(pinecone_index_name)
                    print("[DEBUG] Waiting briefly after delete...")
                    time.sleep(5) # Give Pinecone a moment
                except Exception as e:
                    print(f"[WARN] Failed to delete index '{pinecone_index_name}': {e}. Trying to create anyway.")

            print(f"[DEBUG] Creating Pinecone index '{pinecone_index_name}' (Dim: {dimension}, Metric: {similarity_function})...")
            pc.create_index(
                name=pinecone_index_name,
                dimension=dimension,
                metric=similarity_function,
                spec=ServerlessSpec(cloud="aws", region=pinecone_env) # Region might be optional depending on client version/plan
            )
            # Wait a moment for index to initialize after creation
            print("[DEBUG] Index creation initiated. Waiting briefly...")
            # A short fixed wait. Polling describe_index() until status is 'Ready' is more robust.
            time.sleep(10)
            index = pc.Index(name=pinecone_index_name)
            print(f"[DEBUG] Index '{pinecone_index_name}' assumed ready.")
            upsert_chunks_flag = True # Need to upsert after creating

        elif pinecone_mode == "load":
            if not index_exists:
                print(f"[ERROR] Index '{pinecone_index_name}' does not exist. Cannot load. Aborting.")
                return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

            print(f"[DEBUG] Connecting to existing Pinecone index '{pinecone_index_name}'...")
            index = pc.Index(name=pinecone_index_name)

            # Check index dimension matches data
            stats = index.describe_index_stats()
            if stats.dimension != dimension:
                print(f"[ERROR] Dimension mismatch: Index '{pinecone_index_name}' has dimension {stats.dimension}, but provided data has dimension {dimension}. Aborting.")
                return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
            print(f"[DEBUG] Connected. Index dimension {stats.dimension} matches data.")
            # Decide if upsert happens in load mode. Often, 'load' implies using existing data.
            # Let's assume 'load' means connect *and* potentially upsert new/updated data provided.
            upsert_chunks_flag = True if chunk_objs and chunk_embeddings else False

        else:
            print(f"[ERROR] Invalid pineconeMode '{pinecone_mode}'. Use 'create' or 'load'. Aborting.")
            return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    except Exception as e:
        print(f"[ERROR] Failed during index check/create/load for '{pinecone_index_name}': {e}. Aborting.")
        return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]


    # 4. Upsert embeddings if flagged
    if upsert_chunks_flag and index and chunk_objs and chunk_embeddings:
        print("[DEBUG] Preparing vectors to upsert...")
        vectors_to_upsert = []
        for chunk, embedding in zip(chunk_objs, chunk_embeddings):
            # Ensure a unique string ID for Pinecone
            vector_id = chunk.get("chunkId")
            if not vector_id or not isinstance(vector_id, str):
                 vector_id = str(uuid.uuid4()) # Generate UUID if missing or not string

            metadata = {
                "text": chunk.get("text", ""),
                "docTitle": chunk.get("docTitle", ""),
                "chunkId": vector_id, # Store the ID used in metadata too
            }
            try:
                # Ensure embedding is a list of floats
                embedding_list = [float(x) for x in embedding]
                vectors_to_upsert.append((vector_id, embedding_list, metadata))
            except (TypeError, ValueError) as e:
                 print(f"[WARN] Skipping chunk due to invalid embedding format for ID {vector_id}: {e}")


        if vectors_to_upsert:
            num_to_upsert = len(vectors_to_upsert)
            print(f"[DEBUG] Getting initial vector count in namespace '{pinecone_namespace if pinecone_namespace else '(Default)'}'...")
            initial_count = 0
            try:
                initial_stats = index.describe_index_stats()
                # Handle potential KeyError if namespace doesn't exist yet
                initial_count = initial_stats.namespaces.get(pinecone_namespace, {}).get('vector_count', 0) if pinecone_namespace else initial_stats.total_vector_count
                print(f"[DEBUG] Initial vector count: {initial_count}")
            except Exception as e:
                print(f"[WARN] Could not get initial vector count: {e}. Assuming 0.")

            # Target count needs care if overwriting IDs. A simple sum isn't always right.
            # Checking for increase is better than exact target.
            print(f"[DEBUG] Upserting {num_to_upsert} vectors into namespace '{pinecone_namespace if pinecone_namespace else '(Default)'}'...")
            try:
                # Pinecone client handles internal batching for upsert.
                # For very large datasets (>100k vectors or >2MB payload), consider client-side batching.
                upsert_response = index.upsert(vectors=vectors_to_upsert, namespace=pinecone_namespace)
                print(f"[DEBUG] Upsert call completed. Response: {upsert_response}")

                # --- Smarter Wait Logic ---
                print(f"[DEBUG] Polling index stats (every {polling_interval_seconds}s, max {max_wait_time_seconds}s) waiting for vector count to update...")
                start_time = time.time()
                wait_successful = False
                while True:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_wait_time_seconds:
                        print(f"[WARN] Max wait time ({max_wait_time_seconds}s) exceeded while polling index stats. Proceeding anyway.")
                        break
                    try:
                        current_stats = index.describe_index_stats()
                        current_count = current_stats.namespaces.get(pinecone_namespace, {}).get('vector_count', 0) if pinecone_namespace else current_stats.total_vector_count
                        print(f"[DEBUG] Polling: Current count = {current_count}, Initial = {initial_count}, Time elapsed = {elapsed_time:.1f}s")

                        # Check if count has increased OR if it was non-zero initially (indicating potential overwrite)
                        # This handles cases where initial count was high and we overwrote.
                        if current_count > initial_count or (current_count > 0 and current_count == initial_count and num_to_upsert > 0):
                            print(f"[DEBUG] Vector count updated or stable after upsert ({current_count}). Index likely ready.")
                            wait_successful = True
                            break
                        # Handle case where index was empty and remains empty after upsert (potential issue?)
                        if current_count == 0 and initial_count == 0 and num_to_upsert > 0 and elapsed_time > 15: # Give it 15s
                             print("[WARN] Vector count remains 0 after upsert attempt. Check data or Pinecone status.")
                             # Decide whether to break or keep waiting


                    except Exception as e:
                        print(f"[WARN] Error polling index stats: {e}. Retrying...")

                    time.sleep(polling_interval_seconds)

                print("[DEBUG] Finished waiting/polling.")
                if not wait_successful:
                     print("[WARN] Wait condition might not have been fully met. Retrieval might use slightly stale index.")
                # --- End Smarter Wait Logic ---

            except Exception as e:
                 print(f"[ERROR] Failed during upsert or polling: {e}")
                 # Decide if you want to proceed or abort if upsert/wait fails
                 # Aborting for safety if upsert fails:
                 return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]
        else:
            print("[DEBUG] No valid vectors prepared to upsert.")
    elif upsert_chunks_flag:
         print("[DEBUG] Upsert skipped as no valid chunk objects or embeddings were provided.")


    # 5. Query / Retrieval
    print("[DEBUG] Starting retrieval for queries...")
    if not index:
        print("[ERROR] Index object is not valid. Cannot perform retrieval.")
        return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    for query_obj, query_emb in zip(query_objs, query_embeddings):
        query_text = query_obj.get("text", "N/A") # Get text for logging
        query_short = query_text[:70] + "..." if len(query_text) > 70 else query_text
        print(f"[DEBUG] Processing query: '{query_short}'")
        retrieved_chunks_for_query = []

        try:
            # Ensure query embedding is a list of floats
            query_emb_list = [float(x) for x in query_emb]

            pinecone_response = index.query(
                namespace=pinecone_namespace,
                vector=query_emb_list,
                top_k=top_k,
                include_metadata=True
            )

            # --- Interpret Score and Apply Threshold ---
            print(f"[DEBUG] Raw Pinecone response matches for query '{query_short}': {len(pinecone_response.get('matches', []))}")
            for match in pinecone_response.get("matches", []):
                score = match["score"]
                metadata = match.get("metadata", {})
                chunk_text = metadata.get("text", "")
                doc_title = metadata.get("docTitle", "")
                chunk_id = metadata.get("chunkId", match.id) # Fallback to match ID

                passes_threshold = False
                # Default similarity score is the raw score; adjust if needed
                similarity_score_for_output = score

                # Convert threshold from 0-100 to 0-1 for similarity metrics
                threshold_similarity = raw_similarity_threshold / 100.0

                if similarity_function == "l2":
                     # Lower score (distance) is better.
                     # Convert distance to similarity: 1 / (1 + distance)
                     l2_distance = max(0.0, score) # Ensure non-negative distance
                     similarity_score_for_output = 1.0 / (1.0 + l2_distance)
                     # Compare converted similarity to threshold
                     passes_threshold = (similarity_score_for_output >= threshold_similarity)
                     print(f"  Match(l2): ID={chunk_id}, Dist={score:.4f}, Sim={similarity_score_for_output:.4f}, Threshold={threshold_similarity:.4f}, Passes={passes_threshold}")

                elif similarity_function == "cosine":
                     # Higher score is better. Score is already similarity.
                     # Clamp score to [0, 1] for safety, although Pinecone cosine should be in range.
                     similarity_score_for_output = max(0.0, min(1.0, score))
                     passes_threshold = (similarity_score_for_output >= threshold_similarity)
                     print(f"  Match(cosine): ID={chunk_id}, Score={score:.4f}, Sim={similarity_score_for_output:.4f}, Threshold={threshold_similarity:.4f}, Passes={passes_threshold}")

                elif similarity_function == "dotproduct":
                     # Higher score is better. Score range depends on vectors (not normalized).
                     # Threshold is applied directly to the raw dot product score.
                     similarity_score_for_output = score # Output raw score
                     passes_threshold = (score >= raw_similarity_threshold) # Compare raw score to raw threshold (0-100 interpretation might be wrong here)
                     # Note: User needs to understand dotproduct scale for threshold setting.
                     print(f"  Match(dotproduct): ID={chunk_id}, Score={score:.4f}, RawThreshold={raw_similarity_threshold}, Passes={passes_threshold}")
                else:
                     print(f"[WARN] Unknown similarity function '{similarity_function}' for score interpretation. Passing threshold by default.")
                     passes_threshold = True

                if passes_threshold:
                     retrieved_chunks_for_query.append({
                         "text": chunk_text,
                         "docTitle": doc_title,
                         "chunkId": chunk_id,
                         "similarity": round(similarity_score_for_output, 6) # Standardize output format
                     })

            # Sort final list for this query by similarity score (descending)
            retrieved_chunks_for_query.sort(key=lambda x: x["similarity"], reverse=True)

            # Append result for this query object
            final_results.append({'query_object': query_obj, 'retrieved_chunks': retrieved_chunks_for_query})
            print(f"[DEBUG] Retrieved {len(retrieved_chunks_for_query)} chunks for query: '{query_short}' after filtering/thresholding.")

        except Exception as e:
            print(f"[ERROR] Failed to process query '{query_short}': {e}")
            # Append empty result for this specific query on error
            final_results.append({'query_object': query_obj, 'retrieved_chunks': []})

    print("[DEBUG] Retrieval completed.")
    return final_results

# CHROMADB
import chromadb
import uuid 
import time
from langchain_core.embeddings import Embeddings
from typing import List

@RetrievalMethodRegistry.register("chromadb")
def handle_chromadb(chunk_objs, chunk_embeddings, query_objs, query_embeddings, settings):
    """
    Retrieve chunks using Chroma DB Vector Store with precomputed embeddings.
    Supports in-memory or persistent storage and cosine/l2/ip metrics.
    Aligned with standard handler signature and return structure.
    """
    print("[ChromaDB] Starting retrieval with Chroma...", flush=True)

    # === Step 1: Extract Settings ===
    chroma_mode = settings.get("chromaMode", "memory").lower()  # "memory" or "persistent"
    chroma_path = settings.get("chromaPersistDir", "./chroma_db") # Default path if persistent
    collection_name = settings.get("chromaCollection", f"collection_{uuid.uuid4().hex[:8]}") # Default unique name
    # Chroma uses 'l2', 'cosine', 'ip' (inner product)
    distance_metric = settings.get("chromaDistanceMetric", "cosine").lower()
    top_k = settings.get("top_k", 5)
    # similarity_threshold is 0-100 in settings, convert to 0-1
    raw_similarity_threshold = settings.get("similarity_threshold", 0)
    try:
        similarity_threshold = float(raw_similarity_threshold) / 100.0
        similarity_threshold = max(0.0, min(1.0, similarity_threshold)) # Clamp to [0,1]
    except ValueError:
        print(f"[ChromaDB WARN] Invalid similarity_threshold '{raw_similarity_threshold}'. Defaulting to 0.", flush=True)
        similarity_threshold = 0.0

    cleanup_on_create = settings.get("chromaCleanupOnCreate", True) # bool

    print(f"[ChromaDB] Mode: {chroma_mode}", flush=True)
    print(f"[ChromaDB] Top K: {top_k}", flush=True)
    print(f"[ChromaDB] Similarity threshold: {similarity_threshold:.4f}", flush=True)
    print(f"[ChromaDB] Collection: {collection_name}", flush=True)
    print(f"[ChromaDB] Metric: {distance_metric}", flush=True)
    if chroma_mode == "persistent":
        print(f"[ChromaDB] Persistence path: {chroma_path}", flush=True)
    print(f"[ChromaDB] Cleanup on create: {cleanup_on_create}", flush=True)

    # Consistent result structure initialization
    final_results = []

    # === Basic Input Validation ===
    if not chunk_objs or not chunk_embeddings:
         print("[ChromaDB ERROR] No chunk objects or chunk embeddings provided.", flush=True)
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs] # Consistent error return
    if not query_objs or not query_embeddings:
         print("[ChromaDB ERROR] No query objects or query embeddings provided.", flush=True)
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs] # Consistent error return

    try:
        if not isinstance(chunk_embeddings[0], list) or not isinstance(query_embeddings[0], list):
             raise TypeError("Embeddings should be lists of lists of floats.")
        dimension = len(chunk_embeddings[0])
        query_dimension = len(query_embeddings[0])
        if dimension == 0:
            raise ValueError("Chunk embedding dimension cannot be zero.")
        if query_dimension == 0:
             raise ValueError("Query embedding dimension cannot be zero.")
    except (IndexError, TypeError, ValueError) as e:
         print(f"[ChromaDB ERROR] Invalid embedding structure or dimension: {e}", flush=True)
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    if dimension != query_dimension:
         print(f"[ChromaDB ERROR] Embedding dimension mismatch: Chunks({dimension}), Queries({query_dimension})", flush=True)
         return [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    # === Step 2: Initialize Chroma Client ===
    try:
        print(f"[ChromaDB] Initializing Chroma client (Mode: {chroma_mode})...", flush=True)
        if chroma_mode == "persistent":
            if not os.path.exists(chroma_path):
                 print(f"[ChromaDB] Creating persistence directory: {chroma_path}", flush=True)
                 os.makedirs(chroma_path, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=chroma_path)
        else: # memory mode
            chroma_client = chromadb.Client()
        print("[ChromaDB] Chroma client initialized.", flush=True)

        # === Step 3: Get or Create Collection ===
        print(f"[ChromaDB] Accessing collection: '{collection_name}'", flush=True)

        # Handle cleanup if in create mode and collection exists
        if chroma_mode == "create" and cleanup_on_create:
             try:
                 existing_collections = [col.name for col in chroma_client.list_collections()]
                 if collection_name in existing_collections:
                      print(f"[ChromaDB] 'create' mode: Deleting existing collection '{collection_name}'...", flush=True)
                      chroma_client.delete_collection(name=collection_name)
                      print(f"[ChromaDB] Collection '{collection_name}' deleted.", flush=True)
                 else:
                      print(f"[ChromaDB] 'create' mode: Collection '{collection_name}' does not exist, no need to delete.", flush=True)
             except Exception as e:
                  print(f"[ChromaDB WARN] Failed to delete collection '{collection_name}' during cleanup: {e}. Proceeding...", flush=True)

        # Validate and set metric
        if distance_metric not in ['l2', 'cosine', 'ip']:
            print(f"[ChromaDB WARN] Invalid distance metric '{distance_metric}'. Defaulting to 'cosine'.", flush=True)
            distance_metric = 'cosine'
        collection_metadata = {"hnsw:space": distance_metric}

        print(f"[ChromaDB] Getting or creating collection '{collection_name}' with metric '{distance_metric}'...", flush=True)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata
        )
        print(f"[ChromaDB] Collection '{collection_name}' ready.", flush=True)

        # === Step 4: Add/Upsert Data ===
        print(f"[ChromaDB] Preparing {len(chunk_objs)} items for upsert...", flush=True)
        ids = []
        embeddings_to_add = []
        metadatas_to_add = []
        documents_to_add = [] # Chroma requires text content ('documents')

        for i, chunk in enumerate(chunk_objs):
            chunk_id = chunk.get("chunkId")
            if not chunk_id or not isinstance(chunk_id, str) or len(chunk_id.strip()) == 0:
                 chunk_id = f"chunk_{i}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            try:
                embeddings_to_add.append([float(e) for e in chunk_embeddings[i]])
                metadatas_to_add.append({
                    "docTitle": chunk.get("docTitle", ""),
                    "chunkId": chunk_id # Store original/generated ID
                })
                documents_to_add.append(chunk.get("text", ""))
            except (TypeError, ValueError) as e:
                print(f"[ChromaDB WARN] Skipping chunk ID {chunk_id} due to invalid embedding format: {e}", flush=True)
                # Remove the ID if the data is invalid
                ids.pop()


        if ids:
            print(f"[ChromaDB] Upserting {len(ids)} valid items into collection '{collection_name}'...", flush=True)
            collection.upsert(
                ids=ids,
                embeddings=embeddings_to_add,
                metadatas=metadatas_to_add,
                documents=documents_to_add
            )
            print("[ChromaDB] Upsert operation completed.", flush=True)
        else:
             print("[ChromaDB] No valid items to upsert.", flush=True)


        # === Step 5: Perform Retrieval ===
        print("[ChromaDB] Starting retrieval for queries...", flush=True)

        for query_obj, q_embedding in zip(query_objs, query_embeddings):
            query_text = query_obj.get("text", "N/A")
            query_short = query_text[:70] + "..." if len(query_text) > 70 else query_text
            print(f"[ChromaDB] Processing query: '{query_short}'", flush=True)
            retrieved_chunks_for_query = []

            try:
                query_embedding_float = [float(e) for e in q_embedding]

                query_results = collection.query(
                    query_embeddings=[query_embedding_float],
                    n_results=top_k,
                    include=['metadatas', 'documents', 'distances']
                )

                # === Step 6: Process and Format Results ===
                if query_results and query_results.get('ids') and query_results['ids'][0]:
                    num_results = len(query_results['ids'][0])
                    print(f"[ChromaDB] Query returned {num_results} raw results.", flush=True)

                    for i in range(num_results):
                        distance = query_results['distances'][0][i]
                        metadata = query_results['metadatas'][0][i]
                        doc_text = query_results['documents'][0][i]
                        result_id = query_results['ids'][0][i]

                        similarity_score = 0.0
                        if distance_metric == 'cosine':
                            similarity_score = max(0.0, min(1.0, 1.0 - float(distance)))
                        elif distance_metric == 'l2':
                            similarity_score = 1.0 / (1.0 + float(distance))
                        elif distance_metric == 'ip':
                             # Assume normalized embeddings if IP metric is used for similarity context
                             similarity_score = max(0.0, min(1.0, float(distance)))

                        if similarity_score >= similarity_threshold:
                            retrieved_chunks_for_query.append({
                                "text": doc_text,
                                "similarity": round(similarity_score, 6),
                                "docTitle": metadata.get("docTitle", ""),
                                "chunkId": metadata.get("chunkId", result_id),
                            })
                        else:
                            print(f"[ChromaDB] Result {result_id} skipped due to threshold (Sim: {similarity_score:.4f} < Threshold: {similarity_threshold:.4f})", flush=True)

                    retrieved_chunks_for_query.sort(key=lambda x: x["similarity"], reverse=True)
                    retrieved_chunks_for_query = retrieved_chunks_for_query[:top_k] 

                    final_results.append({
                        'query_object': query_obj,
                        'retrieved_chunks': retrieved_chunks_for_query
                    })
                    print(f"[ChromaDB] Retrieved {len(retrieved_chunks_for_query)} chunks for query '{query_short}' after filtering/sorting.", flush=True)
                else:
                    print(f"[ChromaDB] No results returned from Chroma query for '{query_short}'.", flush=True)
                    final_results.append({'query_object': query_obj, 'retrieved_chunks': []})

            except Exception as e:
                print(f"[ChromaDB ERROR] Failed during query or result processing for query '{query_short}': {e}", flush=True)
                final_results.append({'query_object': query_obj, 'retrieved_chunks': []})

    except Exception as e:
        print(f"[ChromaDB FATAL ERROR] Failed during client/collection setup or upsert: {e}", flush=True)
        final_results = [{'query_object': q_obj, 'retrieved_chunks': []} for q_obj in query_objs]

    print("[ChromaDB] Retrieval process finished.", flush=True)
    return final_results