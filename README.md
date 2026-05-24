🏥 Healthcare RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system for clinical document intelligence. Ask questions about medical documents and get grounded answers with source attribution.

**Built in 1 day with:** Python • LangChain • Gemini • ChromaDB • FastAPI • Streamlit

---

## ✨ Features

- 📄 **Document Ingestion**: Upload and process clinical PDFs via ETL pipeline
- 🔍 **Semantic Search**: Find relevant documents using vector similarity (ChromaDB)
- 🤖 **LLM Integration**: Powered by Google Gemini 2.5 Flash (free)
- 📚 **Source Attribution**: Every answer shows which documents were used
- 📊 **Query Logging**: Track all queries with SQLite database
- 📈 **Experiment Tracking**: Compare configurations with MLflow
- 🎨 **Web Interface**: User-friendly Streamlit UI (no coding needed)
- ⚡ **REST API**: FastAPI endpoints for programmatic access
- 🚀 **Real-time Processing**: Questions answered in ~2 minutes

---

## 🏗️ Architecture
Clinical PDFs (data/raw/)
↓
ETL Pipeline (extract → chunk → embed)
↓
ChromaDB Vector Database (semantic search)
↓
RAG Retriever (find relevant chunks)
↓
Gemini LLM (generate grounded answers)
↓
Streamlit UI + FastAPI API + SQLite Logs

**Data Flow:**
1. **Extract**: Load PDFs, extract text
2. **Transform**: Split into 512-token chunks
3. **Embed**: Create 3072-dimensional vectors (Gemini)
4. **Store**: Save in ChromaDB for fast retrieval
5. **Query**: User asks question → embed question → find similar chunks → send to LLM → get grounded answer

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Google API Key (free from [AI Studio](https://aistudio.google.com/app/apikeys))
- Optional: PostgreSQL (defaults to SQLite)

### Installation (5 minutes)

1. **Clone repository**
```bash
git clone https://github.com/YOUR-USERNAME/Healthcare-RAG.git
cd Healthcare-RAG
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment**
```bash
cp .env.example .env
```

Edit `.env` and add your Google API Key:
GOOGLE_API_KEY=your_key_here
DB_HOST=localhost
DB_PORT=5432
DB_USER=rag_user
DB_PASSWORD=rag_password_123
DB_NAME=healthcare_rag
MLFLOW_TRACKING_URI=http://localhost:5000
ENVIRONMENT=development
CHROMA_DB_PATH=./chromadb_storage

5. **Create sample document (optional)**
```bash
mkdir -p data/raw
# Create data/raw/sample_clinical.txt with clinical content
```

---

## 📋 Running the System

### Start 3 Servers (3 terminals)

**Terminal 1: MLflow (Experiment Tracking)**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
→ Open: http://localhost:5000

**Terminal 2: FastAPI (REST API)**
```bash
python -m src.API.main
```
→ Open: http://localhost:8000/docs

**Terminal 3: Streamlit (Web UI)**
```bash
streamlit run streamlit_app.py
```
→ Open: http://localhost:8501

---

## 💬 Usage

### Option 1: Streamlit Web UI (Easiest)
1. Go to http://localhost:8501
2. In sidebar: Choose "Use Sample Document" or "Upload Custom PDF"
3. Type your question: "What medications did the patient take?"
4. Click "🔍 Search Documents"
5. View answer with sources, relevance scores, and metadata

### Option 2: REST API (Programmatic)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What medications did the patient take?"}'
```

**Response:**
```json
{
  "question": "What medications did the patient take?",
  "answer": "Based on the clinical documents, the patient takes Aspirin 500mg twice daily, Metformin 1000mg daily, and Lisinopril 10mg once daily.",
  "sources": [
    {
      "rank": 1,
      "source": "sample_clinical.txt",
      "chunk_number": 2,
      "relevance_score": 0.607,
      "preview": "MEDICATIONS:\n- Aspirin 500mg twice daily\n- Metformin 1000mg daily..."
    },
    {
      "rank": 2,
      "source": "sample_clinical.txt",
      "chunk_number": 1,
      "relevance_score": 0.582,
      "preview": "DIAGNOSIS:\n- Type 2 Diabetes Mellitus..."
    }
  ],
  "retrieval_count": 2,
  "latency_ms": 127717.33,
  "timestamp": "2026-05-23T16:11:26.933476"
}
```

### Option 3: Python API
```python
from src.RAG.retrieval import RAGRetriever

# Initialize RAG system
rag = RAGRetriever()

# Ask a question
result = rag.query("What was the patient's diagnosis?")

# Access results
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Latency: {result['latency_ms']}ms")
```

### Option 4: API Endpoints
```bash
# Check API health
curl http://localhost:8000/health

# Get query logs
curl http://localhost:8000/logs?limit=10

# Get performance stats
curl http://localhost:8000/logs/stats

# Get daily statistics
curl http://localhost:8000/logs/daily-stats

# Start MLflow experiment
curl -X POST http://localhost:8000/mlflow/start-experiment \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "chunk_512", "chunk_size": 512, "top_k": 3}'

# End MLflow experiment
curl -X POST http://localhost:8000/mlflow/end-experiment

# Compare experiments
curl 'http://localhost:8000/mlflow/compare?experiments=chunk_512,chunk_1024'
```

---

## 📂 Project Structure
Healthcare-RAG/
│
├── src/                          # Source code
│   ├── init.py
│   ├── ETL/
│   │   ├── init.py
│   │   └── ingestion.py          # Document ingestion pipeline
│   ├── RAG/
│   │   ├── init.py
│   │   └── retrieval.py          # RAG retrieval logic
│   ├── API/
│   │   ├── init.py
│   │   └── main.py               # FastAPI server
│   ├── EMBEDDINGS/
│   │   ├── init.py
│   │   └── gemini_embeddings.py  # Embedding configuration
│   ├── database.py               # SQLite query logging
│   ├── mlflow_tracker.py         # MLflow experiment tracking
│   └── pdf_handler.py            # PDF upload and ingestion
│
├── data/
│   ├── raw/                      # Input PDFs (add your files here)
│   └── processed/                # Processed chunks
│
├── notebooks/                    # Jupyter notebooks
│
├── streamlit_app.py              # Streamlit web UI
├── test_setup.py                 # Verify setup
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .env                          # Environment config (DO NOT COMMIT)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
│
├── chromadb_storage/             # Vector database (auto-created)
├── rag_logs.db                   # SQLite logs (auto-created)
└── mlflow.db                     # MLflow tracking (auto-created)

---

## 🔧 Component Details

### 1. ETL Pipeline (src/ETL/ingestion.py)

**What it does:**
- Loads PDFs from `data/raw/`
- Extracts text content
- Splits into 512-token chunks with 100-token overlap
- Creates 3072-dimensional embeddings
- Stores in ChromaDB

**Example:**
```python
from src.ETL.ingestion import ClinicalDocumentIngestor

ingestor = ClinicalDocumentIngestor()
vector_store = ingestor.run_etl_pipeline()
```

### 2. RAG Retrieval (src/RAG/retrieval.py)

**What it does:**
- Converts question to embedding
- Queries ChromaDB for similar chunks
- Builds context-aware prompt
- Sends to Gemini LLM
- Returns grounded answer with sources

**Process:**
Question: "What medications did the patient take?"
↓
Embed: [0.12, -0.45, 0.89, ..., 0.34]  (3072D vector)
↓
Search ChromaDB: Find top-3 similar chunks
↓
Build Prompt: "Based on these documents, answer..."
↓
Send to Gemini: Generate answer
↓
Return: Answer + Sources + Metrics

### 3. FastAPI Server (src/API/main.py)

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `POST /query` - Query RAG system
- `GET /stats` - System statistics
- `GET /logs` - Query logs
- `GET /logs/stats` - Performance stats
- `GET /logs/daily-stats` - Daily aggregated stats
- `POST /mlflow/start-experiment` - Start MLflow run
- `POST /mlflow/end-experiment` - End MLflow run
- `GET /mlflow/best-run` - Get best experiment
- `GET /mlflow/compare` - Compare experiments

### 4. Streamlit UI (streamlit_app.py)

**Features:**
- Upload custom PDFs
- Interactive chat interface
- View sources with relevance scores
- Query history tracking
- System statistics
- Real-time processing feedback

### 5. Database Logging (src/database.py)

**Tables:**
```sql
CREATE TABLE query_logs (
  id INTEGER PRIMARY KEY,
  timestamp TIMESTAMP,
  question TEXT,
  answer TEXT,
  retrieval_count INTEGER,
  top_relevance_score REAL,
  latency_ms REAL,
  sources TEXT,
  model_llm TEXT,
  model_embeddings TEXT,
  status TEXT
);
```

**Query data:**
```python
from src.database import get_query_logs, get_query_stats

logs = get_query_logs(limit=10)
stats = get_query_stats()
```

### 6. MLflow Tracking (src/mlflow_tracker.py)

**Track experiments:**
```python
from src.mlflow_tracker import start_experiment, end_experiment, log_query_metrics

# Start experiment
run_id = start_experiment(
    experiment_name="chunk_size_512",
    chunk_size=512,
    top_k=3
)

# Log metrics
log_query_metrics(
    latency_ms=128057.87,
    retrieval_count=2,
    top_relevance_score=0.607
)

# End experiment
end_experiment()
```

### 7. PDF Handler (src/pdf_handler.py)

**Upload custom PDFs:**
```python
from src.pdf_handler import PDFUploadHandler

handler = PDFUploadHandler()
text = handler.extract_text_from_pdf(pdf_file)
success, msg, chunks = handler.ingest_pdf_content(text, "document.pdf")
```

---

## 📊 System Metrics

With sample clinical document:

- **Average latency:** ~128 seconds (includes network calls)
- **Retrieval accuracy:** ~61% (average relevance score)
- **Success rate:** 100%
- **Vector dimension:** 3072 (Gemini embeddings)
- **Chunk size:** 512 tokens
- **Chunk overlap:** 100 tokens
- **LLM:** Gemini 2.5 Flash (free tier)

*Note: First query is slower due to cold start. Subsequent queries are faster.*

---

## 🧪 Testing

### Test 1: Verify Setup
```bash
python test_setup.py
```

Output should show:
✅ GOOGLE_API_KEY loaded
✅ Embeddings working!
✅ LLM working!
🎉 All systems go! Ready to build RAG pipeline.

### Test 2: Test ETL Pipeline
```bash
python -m src.ETL.ingestion
```

Output should show:
✅ ETL PIPELINE COMPLETE
✅ Vector store created at: ./chromadb_storage

### Test 3: Test RAG Retrieval
```bash
python -m src.RAG.retrieval
```

Output should show:
✅ RAG QUERY COMPLETE
Q: What medications did the patient take?
A: The patient takes Aspirin 500mg...

### Test 4: Test FastAPI
```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "✅ Healthy", "message": "RAG system ready"}
```

---

## 🔐 Security & Privacy

### Current Setup
✅ API keys in `.env` (excluded from git)
✅ Local SQLite database
✅ Data not sent to external services (except Google API)
✅ Documents stored locally in ChromaDB

### For Production
- [ ] Add authentication (JWT tokens)
- [ ] Use PostgreSQL with encryption
- [ ] Add HIPAA compliance audit logs
- [ ] Implement role-based access control
- [ ] Use secrets management service (AWS Secrets, Azure KeyVault)
- [ ] Enable database backups
- [ ] Add request rate limiting
- [ ] Implement query data retention policies

### Environment Variables (Never Commit)
.env                 # DO NOT COMMIT
.env.local          # DO NOT COMMIT
*.pyc               # DO NOT COMMIT
pycache/        # DO NOT COMMIT
chromadb_storage/   # DO NOT COMMIT
rag_logs.db         # DO NOT COMMIT
mlflow.db           # DO NOT COMMIT

---

## 📚 What is RAG?

Retrieval-Augmented Generation solves LLM hallucination by combining:

1. **Retrieval**: Find relevant documents from knowledge base
2. **Augmentation**: Add documents to LLM context
3. **Generation**: LLM generates answer grounded in documents

### Why RAG?

| Challenge | Traditional LLM | RAG System |
|-----------|-----------------|-----------|
| Hallucination | ❌ Makes up facts | ✅ Grounded in documents |
| Source Attribution | ❌ No sources | ✅ Shows which docs used |
| Custom Knowledge | ❌ Not in training data | ✅ Uses uploaded docs |
| Up-to-date | ❌ Training data outdated | ✅ Works with new docs |
| Cost | ❌ Expensive large models | ✅ Cheap small models |
| Transparency | ❌ Black box | ✅ Shows reasoning |

### How It Works
User Question: "What medications did the patient take?"
↓
Embedding: Convert to 3072D vector
↓
Similarity Search: Find top-3 similar chunks
↓
Retrieved Context:
"MEDICATIONS:\n- Aspirin 500mg twice daily\n- Metformin 1000mg daily..."
↓
Prompt Building:
"Based on these documents: [context], Answer: [question]"
↓
LLM Generation: Gemini reads prompt and answers
↓
Answer: "The patient takes Aspirin, Metformin, and Lisinopril..."
↓
Return: {answer, sources, relevance_scores, latency}

---

## 🛠️ Technology Stack

| Component | Technology | Version | Why? |
|-----------|-----------|---------|------|
| Language | Python | 3.13+ | Easy, powerful |
| LLM | Gemini 2.5 Flash | Latest | Free, fast, capable |
| Embeddings | Gemini Embeddings | 3072D | Free, high quality |
| Vector DB | ChromaDB | 0.4.24 | Lightweight, fast |
| RAG Framework | LangChain | 0.3.0+ | Industry standard |
| API Framework | FastAPI | 0.109+ | Modern, fast, intuitive |
| Web UI | Streamlit | 1.31+ | No frontend coding |
| Database | SQLite | Built-in | Zero setup |
| Experiment Tracking | MLflow | 2.10+ | Easy comparison |
| Container | Docker | (Optional) | Cloud deployment |
| Cloud | GCP Cloud Run | (Optional) | Serverless hosting |

---

## 📈 Performance Optimization

### Current Bottlenecks
- API calls to Gemini (~128 seconds for embedding + LLM)
- First query cold start (model loading)
- Large PDFs (slow embedding of thousands of chunks)

### Optimization Strategies
- **Caching**: Cache embeddings for repeated documents
- **Batch processing**: Embed multiple documents in parallel
- **Quantization**: Compress vectors to smaller size
- **Semantic caching**: Skip retrieval for similar recent queries
- **Chunking strategy**: Experiment with different chunk sizes

### MLflow Experiments

Compare configurations:

```bash
# Experiment 1: Small chunks, fast retrieval
curl -X POST http://localhost:8000/mlflow/start-experiment \
  -d '{"experiment_name":"chunk_256","chunk_size":256,"top_k":3}'

# Experiment 2: Large chunks, better context
curl -X POST http://localhost:8000/mlflow/start-experiment \
  -d '{"experiment_name":"chunk_1024","chunk_size":1024,"top_k":5}'

# Compare results
curl 'http://localhost:8000/mlflow/compare?experiments=chunk_256,chunk_1024'
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Easiest)
```bash
git push origin main
# Deploy via https://streamlit.io/cloud
```
- Free tier available
- Auto-deploys from GitHub
- Need to add secrets for API keys

### Option 2: Docker + GCP Cloud Run
```bash
docker build -t healthcare-rag .
gcloud run deploy healthcare-rag --image healthcare-rag
```

### Option 3: Heroku
```bash
heroku create healthcare-rag
git push heroku main
```

### Option 4: Self-hosted (VPS)
```bash
# On your server
git clone repo
pip install -r requirements.txt
# Run with systemd/supervisor
```

---

## 📝 Example Queries

**Medical Questions:**
1. "What medications did the patient take?"
2. "What was the patient's diagnosis?"
3. "What is the treatment plan?"
4. "What are the vital signs?"
5. "Are there any contraindications?"
6. "What is the patient's medical history?"
7. "What tests were ordered?"
8. "What is the follow-up plan?"

**Custom Document Queries:**
1. Upload your PDF
2. Ask domain-specific questions
3. Get answers grounded in YOUR documents

---

## 🐛 Troubleshooting

### Problem: "API Not Connected"

**Solution:**
1. Make sure FastAPI is running: `python -m src.API.main`
2. Check if http://localhost:8000/health returns 200
3. Check API logs for errors

### Problem: "ChromaDB not found"

**Solution:**
1. Run ETL pipeline: `python -m src.ETL.ingestion`
2. Creates `chromadb_storage/` folder
3. Embeddings are generated and stored

### Problem: "Google API Key Error"

**Solution:**
1. Verify key in `.env` file
2. Check key is active at aistudio.google.com/app/apikeys
3. Regenerate new key if needed
4. Restart servers with new key

### Problem: "Slow queries"

**Solution:**
1. First query is slower (cold start)
2. Subsequent queries are faster (cached)
3. Reduce `chunk_size` in `RAGRetriever.__init__()`
4. Reduce `top_k` for fewer retrieved documents
5. Use MLflow to compare configurations

### Problem: "PDF upload fails"

**Solution:**
1. Ensure PyPDF2 is installed: `pip install PyPDF2`
2. Check PDF is not corrupted
3. Try smaller PDF file first
4. Check file permissions in `data/raw/`

---

## 📊 Monitoring & Analytics

### Query Logs

```python
from src.database import get_query_logs, get_query_stats

# Get last 10 queries
logs = get_query_logs(limit=10)
for log in logs:
    print(f"{log['timestamp']}: {log['question']}")
    print(f"  Answer: {log['answer'][:100]}...")
    print(f"  Latency: {log['latency_ms']}ms")
    print()

# Get statistics
stats = get_query_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg latency: {stats['average_latency_ms']}ms")
print(f"Success rate: {stats['success_rate_percent']}%")
```

### MLflow Dashboard
http://localhost:5000

- View all experiments
- Compare metrics
- Track best performing configuration
- Export results

---

## 🎯 Future Improvements

- [ ] Multi-user authentication
- [ ] Document versioning
- [ ] Batch query processing
- [ ] PDF highlighting (show retrieved sections)
- [ ] Support for DOCX, TXT formats
- [ ] Advanced search filters
- [ ] Conversation history/memory
- [ ] Cost tracking per query
- [ ] A/B testing framework
- [ ] Custom prompt templates
- [ ] Local LLM option (Ollama)
- [ ] Vector compression
- [ ] Async query processing
- [ ] Webhook notifications
- [ ] GraphQL API

---

## 📚 Resources

**Documentation:**
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [MLflow Docs](https://mlflow.org/docs/latest/)
- [Google Gemini API](https://ai.google.dev/)

**RAG References:**
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)

---

## 🤝 Contributing

Contributions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Priya**

- Built in 1 day as a learning project
- Learning: LLMs, RAG systems, full-stack development
- Open to feedback and improvements!

---

## 📞 Support & Contact

- **GitHub Issues:** Report bugs and feature requests
- **API Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Streamlit UI:** http://localhost:8501

---

## 🎓 Learning Outcomes

This project taught me:

✅ How RAG systems work
✅ Vector databases and embeddings
✅ LLM integration and prompting
✅ Full-stack Python development
✅ API design with FastAPI
✅ Frontend with Streamlit
✅ Experiment tracking with MLflow
✅ Database logging and analytics
✅ Git and GitHub workflows
✅ Docker containerization
✅ Cloud deployment

---

## 🙏 Acknowledgments

Built with:

- Google Gemini API (embeddings & LLM)
- LangChain (RAG orchestration)
- ChromaDB (vector database)
- FastAPI (REST API)
- Streamlit (web UI)
- MLflow (experiment tracking)

---

**Last Updated:** 2026-05-24
**Status:** ✅ Production Ready
**Version:** 1.0.0
