import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
from contextlib import contextmanager

from flask import Flask, request, jsonify, Response, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, utility, DataType
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.exceptions import BadRequest, InternalServerError, NotFound
import signal
import sys
from threading import Lock
import gc

from config import get_openai_client, MODEL_CONFIG

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    encoding='utf-8',
    force=True,
)
logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration management"""
    MILVUS_HOST = os.environ.get('MILVUS_HOST', 'milvus')
    MILVUS_PORT = int(os.environ.get('MILVUS_PORT', 19530))
    MILVUS_TIMEOUT = int(os.environ.get('MILVUS_TIMEOUT', 30))
    
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
    EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM', 1024))
    
    MAX_TEXT_LENGTH = int(os.environ.get('MAX_TEXT_LENGTH', 8192))
    MAX_QUERY_LENGTH = int(os.environ.get('MAX_QUERY_LENGTH', 512))
    MAX_RESULTS_LIMIT = int(os.environ.get('MAX_RESULTS_LIMIT', 100))
    
    SYSTEM_PROMPT = os.environ.get('SYSTEM_PROMPT', "คุณคือ P'Yui GPT พัฒนาโดยนักศึกษาชั้นปีที่ 4")
    
    # Security and rate limiting
    API_KEY = os.environ.get('API_KEY', '')
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '100 per hour')
    ENABLE_CORS = os.environ.get('ENABLE_CORS', 'false').lower() == 'true'
    
    # Performance
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    CONNECTION_POOL_SIZE = int(os.environ.get('CONNECTION_POOL_SIZE', 10))

class MilvusManager:
    """Thread-safe Milvus connection and collection manager"""
    
    def __init__(self, config: Config):
        self.config = config
        self.collection_name = "document_embeddings"
        self.collection = None
        self._lock = Lock()
        self._connected = False
        
    def connect(self):
        """Establish connection to Milvus with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connections.connect(
                    "default", 
                    host=self.config.MILVUS_HOST, 
                    port=self.config.MILVUS_PORT,
                    timeout=self.config.MILVUS_TIMEOUT
                )
                self._connected = True
                logger.info("Successfully connected to Milvus")
                return
            except Exception as e:
                logger.error(f"Milvus connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def initialize_collection(self):
        """Initialize Milvus collection with error handling"""
        if not self._connected:
            self.connect()
            
        with self._lock:
            try:
                if utility.has_collection(self.collection_name):
                    self.collection = Collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' loaded")
                else:
                    self._create_collection()
                
                # Ensure collection is loaded
                if not self.collection.is_empty:
                    self.collection.load()
                    
            except Exception as e:
                logger.error(f"Failed to initialize collection: {e}")
                raise InternalServerError(f"Database initialization failed: {str(e)}")
    
    def _create_collection(self):
        """Create new Milvus collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.EMBEDDING_DIM),
            FieldSchema(name="created_at", dtype=DataType.INT64),  # Unix timestamp
            FieldSchema(name="metadata", dtype=DataType.JSON)  # For additional document metadata
        ]
        
        schema = CollectionSchema(fields, "Document embeddings collection")
        self.collection = Collection(self.collection_name, schema)
        
        # Create optimized index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("embedding", index_params)
        logger.info(f"Collection '{self.collection_name}' created with HNSW index")
    
    @contextmanager
    def get_collection(self):
        """Context manager for safe collection access"""
        if not self.collection:
            self.initialize_collection()
        yield self.collection

class EmbeddingModel:
    """Thread-safe embedding model wrapper with caching and optimization"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._lock = Lock()
        
    def _load_model(self):
        """Lazy load model and tokenizer"""
        if self._model is None:
            with self._lock:
                if self._model is None:  # Double-check locking
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
                    self._model.eval()  # Set to evaluation mode
                    logger.info("Embedding model loaded successfully")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings with batching and optimization"""
        if not texts:
            return np.array([])
            
        self._load_model()
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch)
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings, dtype=np.float32)
    
    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts"""
        try:
            inputs = self._tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling for better representations
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(1024, dtype=np.float32) for _ in texts]
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize components
config = Config()
milvus_manager = MilvusManager(config)
embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)

# Flask app setup
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Security and rate limiting
if config.ENABLE_CORS:
    CORS(app)

limiter = Limiter(
    key_func=get_remote_address,      # ตั้ง key_func
    default_limits=[config.RATE_LIMIT]  # ตั้งค่า limit
)
limiter.init_app(app)  # ค่อย bind เข้า app

# Metrics tracking
class Metrics:
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.processing_times = []
        self._lock = Lock()
    
    def record_request(self, processing_time: float, error: bool = False):
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            self.processing_times.append(processing_time)
            
            # Keep only recent metrics
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-500:]

metrics = Metrics()

# Decorators
def require_api_key(f):
    """API key authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if config.API_KEY:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Missing or invalid authorization header"}), 401
            
            token = auth_header.split(' ')[1]
            if token != config.API_KEY:
                return jsonify({"error": "Invalid API key"}), 401
                
        return f(*args, **kwargs)
    return decorated_function

def track_metrics(f):
    """Metrics tracking decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        error = False
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error = True
            raise
        finally:
            processing_time = time.time() - start_time
            metrics.record_request(processing_time, error)
    return decorated_function

# Input validation
def validate_text(text: str, max_length: int) -> str:
    """Validate and sanitize text input"""
    if not text or not text.strip():
        raise BadRequest("Text cannot be empty")
    
    text = text.strip()
    if len(text) > max_length:
        raise BadRequest(f"Text too long. Maximum {max_length} characters allowed")
    
    return text

def validate_pagination(limit: Optional[int], offset: Optional[int]) -> Tuple[int, int]:
    """Validate pagination parameters"""
    limit = min(limit or 10, config.MAX_RESULTS_LIMIT)
    offset = max(offset or 0, 0)
    return limit, offset

# Enhanced routes with error handling
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        with milvus_manager.get_collection() as collection:
            # Test database connectivity
            collection.num_entities  # This will raise if connection is bad
            
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "components": {
                "database": "up",
                "embedding_model": "loaded" if embedding_model._model else "not_loaded"
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route("/metrics", methods=["GET"])
@require_api_key
def get_metrics():
    """Get application metrics"""
    avg_processing_time = (
        sum(metrics.processing_times) / len(metrics.processing_times)
        if metrics.processing_times else 0
    )
    
    return jsonify({
        "total_requests": metrics.requests,
        "total_errors": metrics.errors,
        "error_rate": metrics.errors / max(metrics.requests, 1),
        "avg_processing_time": avg_processing_time,
        "uptime": time.time() - app.start_time
    })

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "P'Yui GPT API", "version": "1.0.0"})

@app.route("/index", methods=["POST"])
@require_api_key
@track_metrics
@limiter.limit("50 per hour")
def index_text():
    """Index text with validation and error handling"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("JSON payload required")
            
        text = validate_text(data.get("text"), config.MAX_TEXT_LENGTH)
        metadata = data.get("metadata", {})
        
        # Generate embedding
        embeddings = embedding_model.encode([text])
        if len(embeddings) == 0:
            raise InternalServerError("Failed to generate embedding")
        
        embedding = embeddings[0].tolist()
        
        # Prepare entity
        entity = {
            "text": text,
            "embedding": embedding,
            "created_at": int(time.time()),
            "metadata": metadata
        }
        
        # Insert into Milvus
        with milvus_manager.get_collection() as collection:
            insert_result = collection.insert([entity])
            collection.flush()
            
        logger.info(f"Document indexed successfully: ID {insert_result.primary_keys[0]}")
        
        return jsonify({
            "message": "Text indexed successfully",
            "id": str(insert_result.primary_keys[0]),
            "embedding_dim": len(embedding)
        }), 201
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/search", methods=["POST"])
@require_api_key
@track_metrics
def search_documents():
    """Semantic search with reranking"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("JSON payload required")
            
        query = validate_text(data.get("query"), config.MAX_QUERY_LENGTH)
        top_k = min(data.get("top_k", 5), 20)
        
        # Generate query embedding
        query_embeddings = embedding_model.encode([query])
        if len(query_embeddings) == 0:
            raise InternalServerError("Failed to generate query embedding")
        
        query_embedding = query_embeddings[0].tolist()
        
        # Search in Milvus
        with milvus_manager.get_collection() as collection:
            if collection.num_entities == 0:
                return jsonify({
                    "message": "No documents found",
                    "results": [],
                    "query": query
                })
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 100}
            }
            
            search_results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=min(top_k * 2, 20),  # Retrieve more for reranking
                output_fields=["id", "text", "created_at", "metadata", "embedding"]
            )
            
            # Process and rerank results
            results = []
            document_embeddings = []
            
            for hits in search_results:
                for hit in hits:
                    text = hit.entity.get('text', '')
                    embedding = hit.entity.get('embedding', [])
                    
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='replace')
                    
                    results.append({
                        'id': str(hit.id),
                        'text': text,
                        'score': float(hit.distance),
                        'created_at': hit.entity.get('created_at', 0),
                        'metadata': hit.entity.get('metadata', {})
                    })
                    
                    if embedding:
                        document_embeddings.append(embedding)
            
            # Rerank using cosine similarity
            if document_embeddings and len(document_embeddings) > 1:
                similarities = cosine_similarity([query_embedding], document_embeddings)[0]
                ranked_indices = np.argsort(similarities)[::-1]  # Descending order
                results = [results[i] for i in ranked_indices[:top_k]]
                
                # Update scores with cosine similarity
                for i, result in enumerate(results):
                    result['similarity_score'] = float(similarities[ranked_indices[i]])
            
            return jsonify({
                "message": "Search completed",
                "results": results[:top_k],
                "query": query,
                "total_found": len(results)
            })
            
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/completions", methods=["POST"])
@app.route("/query", methods=["POST"])
@require_api_key
@track_metrics
@limiter.limit("20 per hour")
def completions():
    """Enhanced RAG completions with better error handling"""
    try:
        # Set request encoding to UTF-8
        request.encoding = 'utf-8'
        
        # Get JSON data with explicit UTF-8 encoding
        if request.is_json:
            data = request.get_json(force=True)
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.error(f"JSON decode error: {e}")
                raise BadRequest("Invalid JSON or encoding")

        if not data:
            raise BadRequest("JSON payload required")

        # Extract query with proper UTF-8 handling
        query = data.get("prompt", "")
        if isinstance(query, bytes):
            query = query.decode('utf-8')
        query = validate_text(query, config.MAX_QUERY_LENGTH)

        '''
        {
        "prompt": "สวัสดี",
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": false
        }
        '''

        # Configure logging to use UTF-8
        logger.info(f"User prompt received: {query}")

        stream = data.get("stream", False)
        temperature = max(0.0, min(2.0, data.get("temperature", MODEL_CONFIG.get('temperature', 0.7))))
        max_tokens = min(data.get("max_tokens", MODEL_CONFIG.get('max_tokens', 1000)), 4000)
        top_p = max(0.0, min(1.0, data.get("top_p", MODEL_CONFIG.get('top_p', 0.9))))
        
        # Rest of the function remains the same but ensure UTF-8 encoding
        with milvus_manager.get_collection() as collection:
            if collection.num_entities == 0:
                return Response(
                    json.dumps({
                        "response": "No documents available for search. Please index documents first.",
                        "sources": [],
                        "query": query
                    }, ensure_ascii=False),
                    content_type="application/json; charset=utf-8"
                )
            
            # Retrieve relevant documents
            query_embeddings = embedding_model.encode([query])
            if len(query_embeddings) == 0:
                raise InternalServerError("Failed to generate query embedding")
                
            query_embedding = query_embeddings[0].tolist()
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 100}
            }
            
            search_results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=10,
                output_fields=["id", "text", "embedding", "metadata"]
            )
            
            # Process retrieved documents with UTF-8 handling
            retrieved_docs = []
            doc_embeddings = []
            
            for hits in search_results:
                for hit in hits:
                    entity = hit.entity
                    text = getattr(entity, 'text', '') if entity else ''
                    embedding = getattr(entity, 'embedding', []) if entity else []
                    
                    if isinstance(text, bytes):
                        text = text.decode('utf-8')
                    
                    if text and embedding:
                        retrieved_docs.append({
                            'id': str(hit.id),
                            'text': text,
                            'score': float(hit.distance),
                            'metadata': getattr(entity, 'metadata', {})
                        })
                        doc_embeddings.append(embedding)
            
            # Rerank and select top documents
            if doc_embeddings and len(doc_embeddings) > 1:
                similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                ranked_indices = np.argsort(similarities)[::-1]
                top_docs = [retrieved_docs[i] for i in ranked_indices[:5]]
            else:
                top_docs = retrieved_docs[:5]
            
            if not top_docs:
                return Response(
                    json.dumps({
                        "response": "No relevant documents found for your query.",
                        "sources": [],
                        "query": query
                    }, ensure_ascii=False),
                    content_type="application/json; charset=utf-8"
                )
            
            context = "\n\n".join([doc['text'] for doc in top_docs])
            
            # Prepare messages with UTF-8 encoding
            system_message = f"ช่วยตอบคำถามอย่างชาญฉลาดและแม่นยำ {config.SYSTEM_PROMPT}"
            user_message = f"จากเอกสารต่อไป:\n\n{context}\n\nจงตอบคำถาม: {query}"

            logger.info(f"Prompt: {system_message}\n{user_message}")
            
            # Generate response
            openai_client = get_openai_client()
            response = openai_client.chat.completions.create(
                model=MODEL_CONFIG.get('model', 'typhoon-v2-70b-instruct'),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=MODEL_CONFIG.get('stop'),
                stream=stream
            )
            
            sources = [{"id": doc['id'], "score": doc['score']} for doc in top_docs]
            
            if stream:
                def generate():
                    try:
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                if isinstance(content, bytes):
                                    content = content.decode('utf-8')
                                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        yield f"data: {json.dumps({'error': 'Stream interrupted'}, ensure_ascii=False)}\n\n"
                
                return Response(generate(), mimetype="text/event-stream; charset=utf-8")
            else:
                response_content = response.choices[0].message.content
                if isinstance(response_content, bytes):
                    response_content = response_content.decode('utf-8')
                    
                return Response(
                    json.dumps({
                        "response": response_content,
                        "sources": sources,
                        "query": query
                    }, ensure_ascii=False),
                    content_type="application/json; charset=utf-8"
                )
                
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Completion failed: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/delete/<doc_id>", methods=["DELETE"])
@require_api_key
@track_metrics
def delete_documents(doc_id: str):
    """Delete documents with validation"""
    try:
        with milvus_manager.get_collection() as collection:
            if doc_id == '*':
                # Delete all documents (admin operation)
                delete_result = collection.delete(expr="id >= 0")
                message = "All documents deleted"
            else:
                # Validate doc_id is numeric
                try:
                    int(doc_id)
                except ValueError:
                    raise BadRequest("Document ID must be numeric")
                
                delete_result = collection.delete(expr=f"id == {doc_id}")
                message = f"Document {doc_id} deleted"
            
            collection.flush()
            
            return jsonify({
                "message": message,
                "deleted_count": delete_result.delete_count
            })
            
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/list", methods=["GET"])
@require_api_key
@track_metrics
def list_documents():
    """List documents with pagination and filtering"""
    try:
        query_filter = request.args.get('query', '').strip()
        limit, offset = validate_pagination(
            request.args.get('limit', type=int),
            request.args.get('offset', type=int)
        )
        
        with milvus_manager.get_collection() as collection:
            # Build expression
            expr = f"text like '%{query_filter}%'" if query_filter else None
            
            results = collection.query(
                expr=expr,
                output_fields=["id", "text", "created_at", "metadata"],
                offset=offset,
                limit=limit
            )
            
            documents = []
            for doc in results:
                text = doc.get('text', '')
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')
                
                # Truncate long text for listing
                display_text = text[:200] + "..." if len(text) > 200 else text
                
                documents.append({
                    "id": str(doc['id']),
                    "text": display_text,
                    "created_at": doc.get('created_at', 0),
                    "metadata": doc.get('metadata', {}),
                    "length": len(text)
                })

             # Log the number of documents retrieved
            logger.info(f"Retrieved {len(documents)} documents")
            
            return jsonify({
                "documents": documents,
                "total_returned": len(documents),
                "offset": offset,
                "limit": limit,
                "query_filter": query_filter
            })
            
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error.description)}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized"}), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def rate_limited(error):
    return jsonify({"error": "Rate limit exceeded", "message": str(error.description)}), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Graceful shutdown handling
def signal_handler(sig, frame):
    logger.info("Shutting down gracefully...")
    # Clean up resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Application initialization
def initialize_app():
    """Initialize application components"""
    logger.info("Initializing P'Yui GPT application...")
    
    # Initialize Milvus
    milvus_manager.initialize_collection()
    
    # Pre-load embedding model (optional, for faster first request)
    if os.environ.get('PRELOAD_MODEL', 'false').lower() == 'true':
        logger.info("Pre-loading embedding model...")
        embedding_model.encode(["test"])
    
    app.start_time = time.time()
    logger.info("Application initialized successfully")

if __name__ == "__main__":
    initialize_app()
    
    # Production server configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    if debug:
        logger.warning("Running in DEBUG mode - not suitable for production!")
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        # Use a production WSGI server like Gunicorn in production
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
