# P'Yui GPT

Meet P'Yui GPT - your friendly Thai language assistant! We've created this open-source tool to help make Thai text processing easier and more natural. By bringing together smart databases, advanced language models, and clever search techniques, P'Yui GPT helps you get accurate, relevant answers powered by Typhoon API.

## 🌟 Key Features

- **Vector Database Integration**: Utilizes Milvus for efficient storage and retrieval of document embeddings
- **Multilingual Embedding Model**: Incorporates BAAI/bge-m3 model for high-quality Thai text embeddings
- **Advanced Retrieval**: Two-stage retrieval process with vector search and re-ranking
- **LLM Integration**: Seamless integration with Typhoon API and OpenAI-compatible endpoints
- **RESTful API**: Flask-based web API with rate limiting and metrics
- **Docker Support**: Complete containerized deployment with Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- At least 4GB RAM available
- Port 5001 available on your system

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pyui-gpt.git
   cd pyui-gpt
   ```

2. **Configure API credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Start the services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:5001/health
   ```

The API will be available at `http://localhost:5001`

## 📁 Project Structure

```
pyui-gpt/
├── app/
│   ├── index_docs.py       # Document indexing script
│   ├── query_rag_using_openai.py  # OpenAI client example
│   └── web.py              # Main Flask application
├── docs/                   # Sample documents for indexing
├── .env                    # Environment variables
├── docker-compose.yml      # Container orchestration
├── Dockerfile              # Web service container
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `milvus` | Milvus database host |
| `MILVUS_PORT` | `19530` | Milvus database port |
| `OPENAI_API_KEY` | - | Your Typhoon/OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.opentyphoon.ai/v1` | API base URL |
| `OPENAI_MODEL` | `typhoon-v2.1-12b-instruct` | Model name |
| `SYSTEM_PROMPT` | `คุณคือ P'Yui GPT พัฒนาโดยนักศึกษาชั้นปีที่ 4` | System prompt for responses |

### API Configuration

Edit `.env`:

```python
OPENAI_API_KEY = 'your-api-key-here'
OPENAI_BASE_URL = 'https://api.opentyphoon.ai/v1'
OPENAI_MODEL = 'typhoon-v2.1-12b-instruct'
```

## 📚 API Documentation

### Core Endpoints

#### Health Check
```bash
GET /health
```

#### Index Documents
```bash
POST /index
Content-Type: application/json

{
  "text": "Document content to index",
  "metadata": {
    "title": "Document title",
    "source": "file_upload"
  }
}
```

#### Search Documents
```bash
POST /search
Content-Type: application/json

{
  "query": "Search query",
  "top_k": 5
}
```

#### RAG Completions
```bash
POST /completions
Content-Type: application/json

{
  "prompt": "Your question here",
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

### Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Input question/prompt |
| `max_tokens` | integer | 1000 | Maximum response length (up to 4000) |
| `temperature` | float | 0.7 | Randomness control (0.0-1.0) |
| `top_p` | float | 0.9 | Nucleus sampling (0.0-1.0) |
| `stream` | boolean | false | Enable streaming response |

## 📖 Usage Examples

### Indexing Documents

1. **Prepare documents:**
   - Place `.txt` files in the `docs/` directory
   - Files should follow the document format specified in Document Format section
   - Supported file types: .txt, .md, .pdf
   - Maximum file size: 10MB per document

2. **Document preprocessing:**
   - Documents are automatically split into chunks of configurable size
   - Default chunk size is 1000 characters with 200 character overlap
   - Special characters and formatting are preserved
   - Metadata is extracted from document headers

3. **Run indexing script:**
   ```bash
   python app/index_docs.py
   ```
   This will:
   - Generate embeddings for each document chunk
   - Store vectors in Milvus database
   - Create metadata index for efficient retrieval
   - Log indexing progress and statistics

4. **Verify indexing:**
   ```bash
   curl http://localhost:5001/search \
     -H "Content-Type: application/json" \
     -d '{"query": "test query", "top_k": 1}'
   ```
### Querying P'Yui GPT

#### Using cURL
```bash
curl -X POST http://localhost:5001/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "สวัสดี",
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

#### Using Python OpenAI Client
```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:5001"
)

response = client.chat.completions.create(
    model="typhoon-v2-70b-instruct",
    messages=[
        {"role": "user", "content": "สวัสดี"}
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### Streaming Responses
```bash
curl -X POST http://localhost:5001/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "สวัสดี",
    "stream": true
  }'
```

## 🐳 Container Architecture

| Service | Purpose | Port |
|---------|---------|------|
| **web** | Main Flask API application | 5001 |
| **milvus** | Vector database for embeddings | 19530 |
| **etcd** | Metadata store for Milvus | 2379 |
| **minio** | Object storage for Milvus | 9000 |
| **attu** | Web UI for Milvus Vector Database | 8000 |

## 📊 Monitoring & Metrics

### Health Check
```bash
curl http://localhost:5001/health
```

### Application Metrics
```bash
curl http://localhost:5001/metrics
```

Returns:
- Total requests processed
- Error rates
- Average processing time
- System uptime

## 🔒 Security Features

- **API Key Authentication**: Required for all endpoints
- **Rate Limiting**: 20 requests/hour for completions, 50/hour for indexing
- **Input Validation**: Text length limits and sanitization
- **Error Handling**: Comprehensive error responses

## 🛠️ Development

### Local Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MILVUS_HOST=localhost
export OPENAI_API_KEY=your-key

# Run the application
python app/web.py
```

### Adding New Documents
```bash
# Place .txt files in docs/ directory
# Run indexing script
python app/index_docs.py
```

## 📄 License

Apache 2.0

## 🆘 Troubleshooting

### Common Issues

1. **Port 5001 already in use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "5002:5000"  # Use port 5002 instead
   ```

2. **Milvus connection failed**
   ```bash
   # Check if all containers are running
   docker-compose ps
   
   # Restart services
   docker-compose restart
   ```

3. **Out of memory errors**
   ```bash
   # Increase Docker memory limit to at least 4GB
   # Or reduce max_chunk_size in index_docs.py
   ```

4. **API key errors**
   ```bash
   # Verify your .env has valid credentials
   # Check OPENAI_API_KEY environment variable
   ```

### Getting Help

- Check the health endpoint: `GET /health`
- Review container logs: `docker-compose logs web`
- Verify API credentials in `.env`
- Ensure sufficient system resources (4GB+ RAM)
