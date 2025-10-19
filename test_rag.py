#!/usr/bin/env python3
"""
Script de test pour le système RAG
"""

print("🔍 Test des dépendances RAG...")
print("=" * 50)

# Test 1: Imports de base
print("\n1. Test des imports de base...")
try:
    import streamlit as st
    print("✅ Streamlit")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    import pandas as pd
    print("✅ Pandas")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import numpy as np
    print("✅ NumPy")
except ImportError as e:
    print(f"❌ NumPy: {e}")

# Test 2: Imports RAG
print("\n2. Test des imports RAG...")
try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF (fitz)")
except ImportError as e:
    print(f"❌ PyMuPDF: {e}")

try:
    from tqdm.auto import tqdm
    print("✅ tqdm")
except ImportError as e:
    print(f"❌ tqdm: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✅ langchain_text_splitters")
except ImportError as e:
    print(f"❌ langchain_text_splitters: {e}")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✅ langchain_community.embeddings")
except ImportError as e:
    print(f"❌ langchain_community.embeddings: {e}")

try:
    from langchain_community.vectorstores import FAISS
    print("✅ langchain_community.vectorstores")
except ImportError as e:
    print(f"❌ langchain_community.vectorstores: {e}")

try:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    print("✅ langchain_core.runnables et output_parsers")
except ImportError as e:
    print(f"❌ langchain_core: {e}")

try:
    from langchain_community.llms import HuggingFacePipeline
    print("✅ langchain_community.llms")
except ImportError as e:
    print(f"❌ langchain_community.llms: {e}")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    print("✅ transformers")
except ImportError as e:
    print(f"❌ transformers: {e}")

# Test 3: Test de création d'embeddings
print("\n3. Test de création d'embeddings...")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embeddings créés avec succès")
except Exception as e:
    print(f"❌ Erreur embeddings: {e}")

# Test 4: Test de text splitter
print("\n4. Test de text splitter...")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    test_text = "Ceci est un test de division de texte pour vérifier que le splitter fonctionne correctement."
    chunks = splitter.split_text(test_text)
    print(f"✅ Text splitter fonctionne - {len(chunks)} chunks créés")
except Exception as e:
    print(f"❌ Erreur text splitter: {e}")

# Test 5: Test FAISS
print("\n5. Test FAISS...")
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_texts = ["Test document 1", "Test document 2"]
    db = FAISS.from_texts(test_texts, embeddings)
    print("✅ FAISS créé avec succès")
except Exception as e:
    print(f"❌ Erreur FAISS: {e}")

# Test 6: Test modèle local
print("\n6. Test modèle local...")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
    
    model_name = "google/flan-t5-base"
    print(f"Tentative de chargement du modèle {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Modèle local chargé avec succès")
except Exception as e:
    print(f"❌ Erreur modèle local: {e}")

print("\n" + "=" * 50)
print("🎯 Résumé du diagnostic:")
print("Si tous les tests passent, le problème pourrait être dans l'application Streamlit.")
print("Si certains tests échouent, installez les packages manquants avec:")
print("pip install -r requirements.txt")
