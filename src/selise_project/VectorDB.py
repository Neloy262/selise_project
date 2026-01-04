import os
import hashlib
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus


class VectorStoreManager:
    def __init__(
            self,
            collection_name: str = "rag_collection",
            uri: str = "./milvus_demo.db",
            embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the Milvus vector store manager.
        """
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.collection_name = collection_name
        self.uri = uri

        # Initialize the Milvus client
        self.vector_db = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": self.uri},
            collection_name=self.collection_name,
            # auto_id=True allows Milvus to generate IDs if none are provided,
            # but we will provide them explicitly in add_documents to handle deduplication.
            auto_id=True
        )
        print(f"Vector Store initialized. Connected to: {self.uri}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adds documents to the store.
        Uses MD5 hashing of content to generate deterministic IDs, preventing duplicates.
        """
        if not documents:
            print("No documents provided to add.")
            return []

        print(f"Processing {len(documents)} documents...")

        # Generate deterministic IDs based on page_content
        ids = []
        for doc in documents:
            # Create an MD5 hash of the content
            # If you want metadata to also determine uniqueness, include it in the hash
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            ids.append(content_hash)

        print(f"Adding/Updating documents in Milvus with custom IDs...")

        # We pass the generated 'ids' to add_documents.
        # In Milvus/LangChain, this typically acts as an upsert (update if exists, insert if not).
        result_ids = self.vector_db.add_documents(documents, ids=ids)

        print(f"Successfully processed {len(result_ids)} documents.")
        return result_ids

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Helper to add a single raw text string with optional metadata.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.add_documents([doc])

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a similarity search for the query.
        Returns the top k matching Documents.
        """
        print(f"Searching for: '{query}'")
        results = self.vector_db.similarity_search(query, k=k)
        return results

    def get_retriever(self, k: int = 3):
        """
        Returns a LangChain Retriever interface.
        """
        return self.vector_db.as_retriever(search_kwargs={"k": k})