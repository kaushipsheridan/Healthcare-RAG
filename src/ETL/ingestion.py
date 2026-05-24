"""
ETL Pipeline: Extract, Transform, Load
Ingests clinical documents and prepares them for RAG

Flow:
1. EXTRACT: Load documents from data/raw/
2. TRANSFORM: Split into chunks, embed them
3. LOAD: Store in ChromaDB for retrieval
"""

import os
from pathlib import Path
from typing import List, Dict
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter#Splits text smartly into chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Converts text to vectors using Gemini
from langchain_chroma import Chroma   # Vector DB
from dotenv import load_dotenv         #gets API keys

# Setup logging so we can track what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# WHY CLASS : Think of it as a container that holds:
# Data (variables): chunk_size, embeddings_model, etc.
# Methods (functions): extract_documents(), transform_documents(), etc.
class ClinicalDocumentIngestor:
    """
    Handles ETL pipeline for clinical documents.
    
    Responsibilities:
    - Extract: Load documents from disk
    - Transform: Split into chunks, embed them
    - Load: Store in ChromaDB
    """
    
    def __init__(self):
        """Initialize the ingestor with settings."""
        
        # ===== TEXT SPLITTING SETTINGS =====
        # Why these values?
        # - chunk_size=512: Balance between context and retrieval speed
        #   (512 tokens ≈ 2000 characters)
        # - overlap=100: Keep some context between chunks
        #   (So a fact at chunk boundary isn't lost)

        self.chunk_size = 512 #
        self.chunk_overlap = 100
        
        # ===== EMBEDDING MODEL =====
        # Using Gemini embedding (3072 dimensions)
        self.embeddings_model = "models/gemini-embedding-001"
        
        # ===== PATHS =====
        self.raw_data_path = Path("data/raw")
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chromadb_storage")
        
        # Initialize text splitter
        # RecursiveCharacterTextSplitter is smart:
        # - Tries to split on sentences first
        # - Then on paragraphs
        # - Then on words
        # - Minimizes breaking context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Order of splitting preference
        )
        
        # Initialize embeddings
        # This will use your GOOGLE_API_KEY from .env
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embeddings_model
        )
        
        logger.info("✅ ClinicalDocumentIngestor initialized")
        logger.info(f"   Chunk size: {self.chunk_size}")
        logger.info(f"   Chunk overlap: {self.chunk_overlap}")
        logger.info(f"   Embeddings: {self.embeddings_model}")
    

    
    def extract_documents(self) -> List[Dict[str, str]]:
        """
        EXTRACT PHASE: Load documents from data/raw/
        
        Returns:
            List of dicts with 'content' and 'source' keys
        """
        documents = []
        
        # Check if raw data folder exists
        if not self.raw_data_path.exists():
            logger.warning(f"⚠️  {self.raw_data_path} does not exist!")
            return documents
        
        # Load all .txt and .pdf files
        # (For now, just .txt - pypdf would handle .pdf)
        for file_path in self.raw_data_path.glob("*.txt"):
            try:
                logger.info(f"📄 Loading: {file_path.name}")
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    "content": content,
                    "source": file_path.name,
                    "path": str(file_path)
                })
                
                logger.info(f"   ✅ Loaded {len(content)} characters")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to load {file_path}: {e}")
        
        logger.info(f"\n✅ Extracted {len(documents)} documents total")
        return documents
    
    def transform_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        TRANSFORM PHASE: Split documents into chunks
        
        Why chunking?
        - LLMs have token limits (can't process infinite text)
        - Retrieval is more precise with smaller chunks
        - Embedding quality improves with focused content
        
        Args:
            documents: List of document dicts
            
        Returns:
            List of chunk dicts with 'content', 'source', 'chunk_id'
        """
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            source = doc["source"]
            
            logger.info(f"\n🔪 Chunking: {source}")
            logger.info(f"   Original size: {len(content)} characters")
            
            # Split document into chunks
            # RecursiveCharacterTextSplitter returns list of strings
            split_chunks = self.text_splitter.split_text(content)
            
            logger.info(f"   Split into {len(split_chunks)} chunks")
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(split_chunks):
                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "chunk_id": f"{source}_chunk_{i}",
                    "chunk_number": i,
                    "metadata": {
                        "source": source,
                        "chunk_number": i,
                        "total_chunks": len(split_chunks)
                    }
                })
            
            # Show sample of first chunk
            logger.info(f"   Sample of first chunk (first 200 chars):")
            logger.info(f"   {split_chunks[0][:200]}...")
        
        logger.info(f"\n✅ Created {len(chunks)} chunks total")
        return chunks
    
    def load_to_chromadb(self, chunks: List[Dict[str, str]]) -> Chroma:
        """
        LOAD PHASE: Store chunks and embeddings in ChromaDB
        
        ChromaDB will:
        1. Take each chunk text
        2. Embed it using Gemini embeddings (3072D vectors)
        3. Store both text and vector
        4. Index for fast similarity search
        
        Args:
            chunks: List of chunk dicts
            
        Returns:
            Chroma vector store object
        """
        
        if not chunks:
            logger.error("❌ No chunks to load!")
            return None
        
        # Prepare texts and metadata for ChromaDB
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        
        logger.info(f"\n💾 Loading to ChromaDB...")
        logger.info(f"   Path: {self.chroma_db_path}")
        logger.info(f"   Chunks to embed: {len(texts)}")
        
        try:
            # Create Chroma vector store
            # This will:
            # 1. Embed all texts using Gemini embeddings
            # 2. Store vectors + metadata in ChromaDB
            # 3. Create indices for fast search
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=self.chroma_db_path,
                collection_name="clinical_documents"
            )
            
            logger.info(f"✅ Loaded {len(texts)} chunks into ChromaDB")
            logger.info(f"✅ Vector store created at: {self.chroma_db_path}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to load to ChromaDB: {e}")
            return None
    


    # RUNNING THE PIPELINE (E->T->L)
    def run_etl_pipeline(self) -> Chroma:
        """
        Run the complete ETL pipeline:
        Extract → Transform → Load
        
        Returns:
            Chroma vector store (ready for retrieval)
        """
        
        logger.info("="*60)
        logger.info("🚀 STARTING ETL PIPELINE")
        logger.info("="*60)
        
        # EXTRACT
        documents = self.extract_documents()
        if not documents:
            logger.error("❌ No documents extracted!")
            return None
        
        # TRANSFORM
        chunks = self.transform_documents(documents)
        if not chunks:
            logger.error("❌ No chunks created!")
            return None
        
        # LOAD
        vector_store = self.load_to_chromadb(chunks)
        
        logger.info("\n" + "="*60)
        logger.info("✅ ETL PIPELINE COMPLETE")
        logger.info("="*60)
        
        return vector_store


def main():
    """Run the ETL pipeline."""
    
    ingestor = ClinicalDocumentIngestor()
    vector_store = ingestor.run_etl_pipeline()
    
    if vector_store:
        logger.info("\n✅ Vector store ready for RAG!")
        return vector_store
    else:
        logger.error("\n❌ ETL pipeline failed!")
        return None


if __name__ == "__main__":
    main()