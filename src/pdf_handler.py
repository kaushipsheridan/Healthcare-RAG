"""
PDF Upload and Ingestion Handler
Allows users to upload PDFs and ingest them on-the-fly.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PDFUploadHandler:
    """
    Handles PDF upload, ingestion, and RAG setup.
    """
    
    def __init__(self):
        """Initialize PDF handler."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        logger.info("✅ PDFUploadHandler initialized")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file.
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Extracted text from PDF
        """
        try:
            # Write uploaded file to temporary location
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                dir=tempfile.gettempdir()
            ) as tmp_file:
                tmp_file.write(pdf_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Extract text using PyPDF2
            pdf_reader = PdfReader(tmp_path)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            logger.info(f"✅ Extracted text from PDF ({len(text)} chars)")
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract PDF text: {e}")
            raise


    def ingest_pdf_content(
        self,
        pdf_text: str,
        pdf_name: str,
        collection_name: str = "user_uploads"
    ) -> Tuple[bool, str, int]:
        """
        Ingest PDF text into ChromaDB for RAG.
        
        Args:
            pdf_text: Extracted PDF text
            pdf_name: Name of the PDF
            collection_name: ChromaDB collection to use
            
        Returns:
            Tuple of (success, message, chunk_count)
        """
        try:
            logger.info(f"🔪 Chunking PDF: {pdf_name}")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(pdf_text)
            
            if not chunks:
                return False, "No text extracted from PDF", 0
            
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            texts = chunks
            metadatas = [
                {
                    "source": pdf_name,
                    "chunk_number": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
            
            logger.info(f"💾 Loading to ChromaDB (collection: {collection_name})")
            
            # Create ChromaDB vector store
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=f"./chromadb_uploads_{collection_name}",
                collection_name=collection_name
            )
            
            logger.info(f"✅ Successfully ingested PDF into ChromaDB")
            
            return True, f"Successfully ingested {len(chunks)} chunks from {pdf_name}", len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest PDF: {e}")
            return False, f"Error ingesting PDF: {str(e)}", 0


    def get_vector_store(self, collection_name: str = "user_uploads"):
        """
        Get ChromaDB vector store for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Chroma vector store
        """
        try:
            vector_store = Chroma(
                persist_directory=f"./chromadb_uploads_{collection_name}",
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            return vector_store
        except Exception as e:
            logger.error(f"❌ Failed to get vector store: {e}")
            return None