"""
RAG Retrieval Module
Handles: Vector similarity search, prompt building, LLM inference
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RAGRetriever:
    """
    Retrieval-Augmented Generation system.
    
    Flow:
    1. User asks question
    2. Embed question (same space as documents)
    3. Query ChromaDB for similar chunks
    4. Build prompt with retrieved context
    5. Send to LLM for grounded answer
    6. Return answer + sources
    """
    
    def __init__(self, chroma_db_path: str = "./chromadb_storage"):
        """
        Initialize RAG system.
        
        Args:
            chroma_db_path: Path to ChromaDB directory
        """
        
        logger.info("🚀 Initializing RAG Retriever...")
        
        # Initialize embeddings (same model as ingestion)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        # Load existing ChromaDB vector store
        # This loads the embeddings we created in ingestion.py
        try:
            self.vector_store = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=self.embeddings,
                collection_name="clinical_documents"
            )
            logger.info(f"✅ Loaded ChromaDB from {chroma_db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load ChromaDB: {e}")
            logger.error("   Make sure you ran ingestion.py first!")
            self.vector_store = None
        
        # Initialize LLM (Gemini 2.5 Flash)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # RAG settings
        self.top_k = 3  # How many chunks to retrieve
        self.max_context_length = 2000  # Max characters to send to LLM
        
        logger.info(f"✅ RAG Retriever ready!")
        logger.info(f"   Top-K: {self.top_k}")
        logger.info(f"   Max context length: {self.max_context_length}")
    
    def retrieve_relevant_chunks(self, question: str) -> List[Dict]:
        """
        STEP 1: Query ChromaDB for relevant chunks.
        
        What happens:
        1. Convert question to vector (3072D)
        2. Find top-K most similar chunk vectors
        3. Return chunks + metadata
        
        Args:
            question: User's question
            
        Returns:
            List of relevant chunks with metadata
        """
        
        logger.info(f"\n🔍 Retrieving chunks for: '{question}'")
        
        if not self.vector_store:
            logger.error("❌ Vector store not initialized!")
            return []
        
        try:
            # Query ChromaDB using similarity search
            # This:
            # 1. Embeds the question
            # 2. Calculates similarity to all chunks
            # 3. Returns top_k most similar
            results = self.vector_store.similarity_search_with_relevance_scores(
                question,
                k=self.top_k
            )
            
            logger.info(f"✅ Retrieved {len(results)} chunks")
            
            # Format results
            chunks = []
            for i, (doc, score) in enumerate(results):
                chunk_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "rank": i + 1
                }
                chunks.append(chunk_dict)
                logger.info(f"   [{i+1}] Score: {score:.2f}, Source: {doc.metadata.get('source', 'unknown')}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        STEP 2: Build prompt with retrieved context.
        
        Creates a prompt that:
        - Provides system instructions
        - Includes retrieved documents
        - Contains the user question
        - Instructs LLM to ground answer in documents
        
        Args:
            question: Original user question
            retrieved_chunks: Retrieved relevant chunks
            
        Returns:
            Formatted prompt string
        """
        
        logger.info("\n📝 Building prompt...")
        
        if not retrieved_chunks:
            logger.warning("⚠️  No chunks to build prompt from!")
            return f"Question: {question}\n\nAnswer: No relevant documents found."
        
        # Build context section
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"--- DOCUMENT {i+1} (from {chunk['metadata'].get('source', 'unknown')}, chunk {chunk['metadata'].get('chunk_number', 'unknown')}) ---\n"
                f"{chunk['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            logger.warning(f"⚠️  Context too long ({len(context)} chars), truncating...")
            context = context[:self.max_context_length] + "\n[Context truncated...]"
        
        # Build full prompt
        # This is a "few-shot" prompt with instructions
        prompt = f"""You are a medical information assistant. Your job is to answer questions based ONLY on the provided clinical documents.

IMPORTANT INSTRUCTIONS:
1. Answer only using information from the documents below
2. If the answer is not in the documents, say "This information is not available in the provided documents."
3. Be precise and cite which document you're referencing
4. Do not make assumptions or use external medical knowledge
5. If documents conflict, acknowledge both perspectives

CLINICAL DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
        
        logger.info(f"✅ Prompt built ({len(prompt)} chars)")
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        STEP 3: Send prompt to LLM and get answer.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            LLM-generated answer
        """
        
        logger.info("\n🤖 Generating answer with Gemini...")
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            logger.info(f"✅ Answer generated ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"❌ LLM failed: {e}")
            return "Error generating answer. Please try again."
    
    def query(self, question: str) -> Dict:
        """
        MAIN RAG QUERY FUNCTION
        
        Complete flow:
        1. Retrieve relevant chunks from ChromaDB
        2. Build prompt with context
        3. Generate answer with LLM
        4. Format response with sources
        
        Args:
            question: User's question
            
        Returns:
            Dict with answer, sources, metadata
        """
        
        logger.info("="*60)
        logger.info(f"📌 RAG QUERY START")
        logger.info(f"Question: {question}")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # STEP 1: Retrieve
        retrieved_chunks = self.retrieve_relevant_chunks(question)
        
        # STEP 2: Build Prompt
        prompt = self.build_prompt(question, retrieved_chunks)
        
        # STEP 3: Generate Answer
        answer = self.generate_answer(prompt)
        
        # STEP 4: Format response with sources
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        response = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "rank": chunk["rank"],
                    "source": chunk["metadata"].get("source", "unknown"),
                    "chunk_number": chunk["metadata"].get("chunk_number", "unknown"),
                    "relevance_score": chunk["relevance_score"],
                    "preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                }
                for chunk in retrieved_chunks
            ],
            "retrieval_count": len(retrieved_chunks),
            "latency_ms": round(latency_ms, 2),
            "timestamp": start_time.isoformat()
        }
        
        logger.info("="*60)
        logger.info(f"✅ RAG QUERY COMPLETE")
        logger.info(f"Latency: {latency_ms:.2f}ms")
        logger.info(f"Sources used: {len(retrieved_chunks)}")
        logger.info("="*60)
        
        return response


def main():
    """Test RAG retriever."""
    
    # Initialize RAG
    rag = RAGRetriever()
    
    if not rag.vector_store:
        logger.error("❌ Cannot test - ChromaDB not loaded!")
        logger.error("   Run ingestion.py first: python -m src.ETL.ingestion")
        return
    
    # Test queries
    test_questions = [
        "What medications did the patient take?",
        "What was the patient's diagnosis?",
        "What is the treatment plan?",
    ]
    
    for question in test_questions:
        result = rag.query(question)
        
        print("\n" + "="*60)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({result['retrieval_count']}):")
        for source in result['sources']:
            print(f"  [{source['rank']}] {source['source']} (chunk {source['chunk_number']}) - Score: {source['relevance_score']:.2f}")
        print(f"\nLatency: {result['latency_ms']:.2f}ms")
        print("="*60)


if __name__ == "__main__":
    main()