"""
Test script to verify our setup is correct.
Tests:
1. Environment variables load
2. Gemini Embeddings work
3. Gemini LLM works
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Verify API key exists
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "paste_your_actual_google_api_key_here":
    print("❌ GOOGLE_API_KEY not found or not set in .env!")
    print("   Please add your Google API key to .env")
    exit(1)
else:
    print("✅ GOOGLE_API_KEY loaded")

# === Test 1: Embeddings ===
print("\n--- Testing Embeddings ---")
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Embed a simple sentence
    test_text = "The patient has diabetes and hypertension."
    embedding_vector = embeddings.embed_query(test_text)
    
    print(f"✅ Embeddings working!")
    print(f"   Text: '{test_text}'")
    print(f"   Vector dimension: {len(embedding_vector)}")
    print(f"   First 5 values: {embedding_vector[:5]}")
    
except Exception as e:
    print(f"❌ Embeddings failed: {e}")
    exit(1)

# === Test 2: LLM ===
print("\n--- Testing LLM ---")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # Send a simple prompt
    response = llm.invoke("Say 'Setup successful!' and nothing else.")
    
    print(f"✅ LLM working!")
    print(f"   Response: {response.content}")
    
except Exception as e:
    print(f"❌ LLM failed: {e}")
    exit(1)

print("\n" + "="*50)
print("🎉 All systems go! Ready to build RAG pipeline.")
print("="*50)