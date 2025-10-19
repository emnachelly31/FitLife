#!/usr/bin/env python3
"""
Test simple du système RAG
"""

def test_rag_imports():
    """Test des imports RAG"""
    try:
        import fitz
        from tqdm.auto import tqdm
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        
        print("✅ Tous les imports RAG fonctionnent !")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_rag_system():
    """Test de création d'un système RAG simple"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # Texte de test
        test_text = """
        La nutrition est l'ensemble des processus par lesquels un organisme vivant utilise les aliments pour assurer son fonctionnement.
        Les macronutriments comprennent les protéines, les glucides et les lipides.
        Les micronutriments comprennent les vitamines et les minéraux.
        Une alimentation équilibrée est essentielle pour la santé.
        """
        
        # Diviser le texte
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = splitter.split_text(test_text)
        
        # Créer les embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Créer la base vectorielle
        db = FAISS.from_texts(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})
        
        # Test de recherche
        docs = retriever.invoke("Qu'est-ce que la nutrition ?")
        
        print("✅ Système RAG créé avec succès !")
        print(f"✅ Recherche fonctionnelle - {len(docs)} documents trouvés")
        
        return True
    except Exception as e:
        print(f"❌ Erreur système RAG: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test du système RAG...")
    print("=" * 40)
    
    # Test 1: Imports
    imports_ok = test_rag_imports()
    
    if imports_ok:
        # Test 2: Système RAG
        test_rag_system()
    
    print("=" * 40)
    print("🎯 Test terminé !")
