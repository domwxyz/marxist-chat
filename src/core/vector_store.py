import shutil
from pathlib import Path
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
import unicodedata

from core.llm_manager import LLMManager
import config

class VectorStoreManager:
    def __init__(self, vector_store_dir=None, cache_dir=None):
        self.vector_store_dir = vector_store_dir or config.VECTOR_STORE_DIR
        self.cache_dir = cache_dir or config.CACHE_DIR
    
    def create_vector_store(self, overwrite=False):
        """Create a new vector store from cached documents"""
        # Check if cache directory exists and has content
        cache_dir_has_content = False
        if self.cache_dir.exists():
            cache_dir_has_content = any(self.cache_dir.glob("*.txt"))
            
        if not cache_dir_has_content:
            print("\nError: No RSS archive found or archive is empty. Please archive RSS feed first.")
            return None
        
        # Check if vector store exists and has content
        vector_store_exists = self.vector_store_dir.exists() and any(self.vector_store_dir.iterdir())
            
        if vector_store_exists:
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
        
        # Set the node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=25,
            paragraph_separator="\n\n"
        )
        
        print("Embedding model initialized and settings configured.")
        
        try:
            self.vector_store_dir.mkdir(exist_ok=True, parents=True)
            
            # Use default storage context
            storage_context = StorageContext.from_defaults()
            
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
            
            if not valid_documents:
                print("\nError: No valid documents found in the RSS archive.")
                return None
                
            print(f"\nProcessing {len(valid_documents)} valid documents...")
            
            # Create index from documents
            index = VectorStoreIndex.from_documents(
                valid_documents,
                storage_context=storage_context,
                show_progress=True,
                service_context=Settings
            )
            
            print(f"\nSaving vector store at {self.vector_store_dir}")
            index.storage_context.persist(persist_dir=self.vector_store_dir)
            print("\nVector store created successfully!")
            
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
            
            # Load storage context
            storage_context = StorageContext.from_defaults(persist_dir=self.vector_store_dir)
            
            # Load index
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=Settings
            )
            
            print("Vector store loaded successfully!")
            return index
            
        except Exception as e:
            print(f"\nError loading vector store: {e}")
            return None
