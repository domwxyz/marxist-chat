import shutil
from pathlib import Path
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import unicodedata

from core.llm_manager import LLMManager
import config

class VectorStoreManager:
    def __init__(self, vector_store_dir=None, cache_dir=None):
        self.vector_store_dir = vector_store_dir or config.VECTOR_STORE_DIR
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.chroma_client = None
        self.chroma_collection = None
    
    def _init_chroma_client(self):
        """Initialize ChromaDB client and collection"""
        # Ensure vector store directory exists
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence path
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        
        # Create or get collection named "articles"
        self.chroma_collection = self.chroma_client.get_or_create_collection("articles")
        
        return self.chroma_collection
    
    def create_vector_store(self, overwrite=False):
        """Create a new vector store from cached documents"""
        if not self.cache_dir.exists() or not any(self.cache_dir.iterdir()):
            print("\nError: No RSS archive found. Please archive RSS feed first.")
            return None
            
        if self.vector_store_dir.exists():
            if not overwrite:
                print("\nVector store already exists. Set overwrite=True to recreate it.")
                return None
            try:
                shutil.rmtree(self.vector_store_dir)
            except Exception as e:
                print(f"\nError deleting existing vector store: {e}")
                return None
                
        print("\nInitializing embedding model...")
        embed_model = LLMManager.initialize_embedding_model()
        
        # Set the embedding model in Settings before creating the index
        Settings.embed_model = embed_model
        
        # Set the node parser for chunking documents
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=25,
            paragraph_separator="\n\n"
        )
        
        print("Embedding model initialized and settings configured.")
        
        try:
            # Initialize ChromaDB
            chroma_collection = self._init_chroma_client()
            
            # Create a vector store using ChromaDB
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create storage context with ChromaDB vector store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Process all documents
            print("Loading documents...")
            all_documents = []
            total_files = len(list(self.cache_dir.glob("*.txt")))
            
            for i, file_path in enumerate(self.cache_dir.glob("*.txt"), 1):
                try:
                    if file_path.stat().st_size == 0:
                        continue
                        
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                        
                    if not text.strip():
                        continue
                        
                    # Apply consistent normalization for all text content
                    text = unicodedata.normalize('NFKC', text)
                    
                    from llama_index.core import Document
                    all_documents.append(Document(text=text))
                    
                    if i % 100 == 0:
                        print(f"Loaded {i}/{total_files} documents...")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            
            # Filter valid documents
            valid_documents = [doc for doc in all_documents if len(doc.text.strip()) > 50]
            print(f"\nProcessing {len(valid_documents)} valid documents...")
            
            # Create index from documents
            index = VectorStoreIndex.from_documents(
                valid_documents,
                storage_context=storage_context,
                show_progress=True,
                service_context=Settings
            )
            
            print(f"\nVector store created successfully at {self.vector_store_dir}")
            return index
            
        except Exception as e:
            print(f"\nError creating vector store: {e}")
            if self.vector_store_dir.exists():
                try:
                    shutil.rmtree(self.vector_store_dir)
                except Exception:
                    pass
            return None
    
    def load_vector_store(self):
        """Load the existing vector store"""
        print(f"Attempting to load vector store from: {self.vector_store_dir}")
        
        if not self.vector_store_dir.exists():
            print(f"\nError: No vector store found at {self.vector_store_dir}")
            return None
            
        try:
            # Initialize LLM and embedding model
            llm = LLMManager.initialize_llm()
            embed_model = LLMManager.initialize_embedding_model()
            
            # Configure settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=25,
                paragraph_separator="\n\n"
            )
            
            # Initialize ChromaDB client
            print("Initializing ChromaDB client...")
            chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            collection_name = "articles"
            
            # Check if the collection exists
            collection_names = chroma_client.list_collections()
            print(f"Found collections: {collection_names}")
            
            if collection_name not in collection_names:
                print(f"No collection named '{collection_name}' found in ChromaDB")
                return None
                
            # Get the collection
            chroma_collection = chroma_client.get_collection(collection_name)
            
            # Create vector store with the ChromaDB collection
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create a new index with the existing vector store
            print("Creating index from the ChromaDB vector store...")
            from llama_index.core import VectorStoreIndex
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            print("Vector store loaded successfully!")
            return index
            
        except Exception as e:
            print(f"\nError loading vector store: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_document_by_id(self, document_id: str):
        """Retrieve a document by ID (filename)"""
        # In our implementation, document_id would be the filename
        file_path = self.cache_dir / f"{document_id}.txt"
        
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                
            # Extract metadata from text
            metadata = self._extract_metadata_from_text(text)
            metadata["file_name"] = f"{document_id}.txt"
            
            from llama_index.core import Document
            return Document(text=text, metadata=metadata)
        except Exception as e:
            print(f"Error loading document {document_id}: {str(e)}")
            return None

    def search_documents(self, query: str, limit: int = 10):
        """Search for documents similar to the query text"""
        # Initialize vector store if needed
        if not self._vector_store_loaded():
            self.load_vector_store()
            
        if not self.chroma_collection:
            self._init_chroma_client()
            
        try:
            # Import embedding model for query embedding
            from core.llm_manager import LLMManager
            embed_model = LLMManager.initialize_embedding_model()
            
            # Get query embedding
            query_embedding = embed_model.get_text_embedding(query)
            
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            documents_with_scores = []
            
            if results and "documents" in results and results["documents"]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if "metadatas" in results else {}
                    distance = results["distances"][0][i] if "distances" in results else 1.0
                    
                    # Convert distance to similarity score (1 - distance) and normalize
                    score = 1.0 - min(1.0, distance)
                    
                    # Create document object
                    from llama_index.core import Document
                    doc = Document(text=doc_text, metadata=metadata)
                    documents_with_scores.append((doc, score))
                    
            return documents_with_scores
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def _extract_metadata_from_text(self, text: str):
        """Extract metadata section from document text"""
        metadata = {}
        lines = text.split("\n")
        
        in_metadata = False
        for line in lines:
            if line.strip() == "---":
                if not in_metadata:
                    in_metadata = True
                    continue
                else:
                    break
                    
            if in_metadata and ": " in line:
                key, value = line.split(": ", 1)
                metadata[key] = value
                
        return metadata

    def _vector_store_loaded(self):
        """Check if vector store is loaded"""
        return self.chroma_client is not None and self.chroma_collection is not None
        
    def test_search(self, query="communism"):
        """Test searching in the vector store"""
        try:
            if not self.chroma_collection:
                self._init_chroma_client()
            
            # Get embedding model
            from core.llm_manager import LLMManager
            embed_model = LLMManager.initialize_embedding_model()
            
            # Search directly in the collection
            print(f"DEBUG: Testing direct search for '{query}'")
            query_embedding = embed_model.get_text_embedding(query)
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            print(f"DEBUG: Search results: {results}")
            return results
        except Exception as e:
            print(f"ERROR: Search test failed: {e}")
            import traceback
            print(traceback.format_exc())
            return None