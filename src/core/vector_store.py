import shutil
from pathlib import Path
from llama_index.core import Settings, Document, VectorStoreIndex, load_index_from_storage, StorageContext

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
import chromadb
import unicodedata

from core.feed_processor import FeedProcessor
from core.llm_manager import LLMManager
from core.metadata_repository import MetadataRepository
import config

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

class VectorStoreManager:
    def __init__(self, vector_store_dir=None, cache_dir=None):
        self.vector_store_dir = vector_store_dir or config.VECTOR_STORE_DIR
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.chroma_client = None
        self.chroma_collection = None
        self.metadata_repository = MetadataRepository(self.cache_dir)
        self.feed_processor = FeedProcessor()  # Used for feed directory name generation
    
    def _init_chroma_client(self):
        """Initialize ChromaDB client and collection"""
        # Ensure vector store directory exists
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence path
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        
        # Create or get collection named "articles"
        self.chroma_collection = self.chroma_client.get_or_create_collection("articles")
        
        return self.chroma_collection

    def _get_all_document_paths(self):
        """Get paths to all documents across all feed directories"""
        all_paths = []
        
        # First check if there are any documents directly in the cache directory (old structure)
        direct_files = list(self.cache_dir.glob("*.txt"))
        if direct_files:
            all_paths.extend(direct_files)
            
        # Then check all subdirectories (new structure)
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                subdir_files = list(subdir.glob("*.txt"))
                all_paths.extend(subdir_files)
                
        return all_paths
    
    def create_vector_store(self, overwrite=False):
        """Create a new vector store from cached documents with improved metadata preservation"""
        all_document_paths = self._get_all_document_paths()
        
        if not all_document_paths:
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

        try:
            embed_model = LLMManager.initialize_embedding_model()
            print("Embedding model initialized successfully")
            
            # Set the embedding model in Settings
            Settings.embed_model = embed_model
            
            Settings.node_parser = SentenceSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            
            print("Node parser initialized successfully")
        except Exception as e:
            print(f"\nERROR initializing models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
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
            total_files = len(all_document_paths)

            for i, file_path in enumerate(all_document_paths, 1):
                try:
                    if file_path.stat().st_size == 0:
                        continue
                        
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                        
                    if not text.strip():
                        continue
                    
                    # Extract metadata from text
                    metadata = self._extract_metadata_from_text(text)
                    
                    # Ensure filename is in metadata
                    metadata['file_name'] = file_path.name
                    
                    # Identify the source feed directory
                    relative_path = file_path.relative_to(self.cache_dir)
                    parts = relative_path.parts
                    if len(parts) > 1:  # If file is in a subdirectory
                        metadata['feed_name'] = parts[0]  # First part is the feed directory name
                    else:
                        metadata['feed_name'] = 'unknown'  # For files directly in cache dir
                    
                    # Restructure the text to emphasize metadata
                    # This makes it more likely to be captured within each chunk
                    enhanced_text = f"Title: {metadata.get('title', 'Untitled')}\n"
                    enhanced_text += f"Author: {metadata.get('author', 'Unknown')}\n"
                    enhanced_text += f"Feed Source: {metadata.get('feed_name', 'Unknown')}\n"
                    
                    if 'categories' in metadata:
                        enhanced_text += f"Categories: {metadata.get('categories', '')}\n"
                        
                    enhanced_text += f"Date: {metadata.get('date', 'Unknown')}\n"
                    enhanced_text += f"File: {metadata.get('file_name', '')}\n\n"
                    
                    # Append the original text
                    enhanced_text += text
                        
                    # Apply consistent normalization for all text content
                    enhanced_text = unicodedata.normalize('NFKC', enhanced_text)
                    
                    all_documents.append(Document(text=enhanced_text, metadata=metadata))
                    
                    if i % 100 == 0:
                        print(f"Loaded {i}/{total_files} documents...")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            
            # Filter valid documents
            valid_documents = [doc for doc in all_documents if len(doc.text.strip()) > 50]
            print(f"\nProcessing {len(valid_documents)} valid documents...")
            
            # Create the index with standard chunking
            index = VectorStoreIndex.from_documents(
                valid_documents,
                storage_context=storage_context,
                show_progress=True,
                service_context=Settings
            )
            
            # Post-process: explicitly copy metadata to each node in the index
            print("Post-processing nodes to ensure metadata is preserved...")
            try:
                # Access the nodes in the index and ensure metadata is preserved
                for node_id, node in index.docstore.docs.items():
                    if hasattr(node, 'metadata'):
                        # Extract document ID from node
                        doc_id = node.ref_doc_id
                        
                        # Try to find the source document metadata
                        file_name = node.metadata.get('file_name', '')
                        source_doc = None
                        
                        # Try to find the source document by file_name if it's not in node metadata
                        if not file_name:
                            for doc in valid_documents:
                                if doc.doc_id == doc_id:
                                    source_doc = doc
                                    break
                        
                        # If we found the source document, copy its metadata
                        if source_doc and hasattr(source_doc, 'metadata'):
                            for key in ['file_name', 'title', 'date', 'author', 'url', 'categories', 'feed_name']:
                                if key in source_doc.metadata and key not in node.metadata:
                                    node.metadata[key] = source_doc.metadata[key]
                        
                        # Add chunk_id to metadata if file_name is available
                        if 'file_name' in node.metadata:
                            node.metadata['chunk_id'] = f"{node.metadata['file_name']}:{node.start_char_idx}-{node.end_char_idx}"
            except Exception as e:
                print(f"Warning: Error during node post-processing: {e}")
                # Continue anyway - this is a best-effort enhancement
            
            # Build metadata index
            print("Building metadata index...")
            self.metadata_repository.build_metadata_index(force_rebuild=True)
            
            print(f"\nVector store created successfully at {self.vector_store_dir}")
            return index
            
        except Exception as e:
            print(f"\nError creating vector store: {e}")
            import traceback
            traceback.print_exc()
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
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                paragraph_separator="\n\n"
            )
            
            # Initialize ChromaDB client
            print("Initializing ChromaDB client...")
            self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            collection_name = "articles"

            # Check if the collection exists
            collection_names = self.chroma_client.list_collections()  # Now directly returns names
            print(f"Found collections: {collection_names}")

            if not collection_names or collection_name not in collection_names:
                print(f"No collection named '{collection_name}' found in ChromaDB")
                return None

            # Get the collection
            self.chroma_collection = self.chroma_client.get_collection(collection_name)

            vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Create a new index with the existing vector store
            print("Creating index from the ChromaDB vector store...")
            index = VectorStoreIndex.from_vector_store(vector_store)
            
            # Load metadata repository
            self.metadata_repository.load_metadata_index()
            
            print("Vector store loaded successfully!")
            return index
            
        except Exception as e:
            print(f"\nError loading vector store: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def update_vector_store(self):
        """Update the vector store with new documents without rebuilding"""
        # Check if vector store exists
        if not self.vector_store_dir.exists():
            print("\nError: No vector store found. Please create one first.")
            return False
            
        try:
            # Get the latest document date for each feed source
            latest_dates = self.get_latest_document_dates_by_feed()
            if not latest_dates:
                print("\nNo existing documents found. You should create the vector store instead.")
                return False
                
            # Initialize FeedProcessor
            feed_processor = FeedProcessor()
            feed_urls = config.RSS_FEED_URLS
            
            all_new_documents = []
            
            # Process each feed individually to check for new content
            for feed_url in feed_urls:
                feed_dir_name = feed_processor._get_feed_directory_name(feed_url)
                latest_date = latest_dates.get(feed_dir_name, None)
                
                if latest_date:
                    print(f"\nLooking for documents newer than {latest_date} from {feed_url}")
                else:
                    print(f"\nNo existing documents found for {feed_url}, will fetch all entries")
                
                # For each feed, fetch entries since its specific latest date
                if latest_date:
                    # Only fetch entries for this specific feed
                    entries = []
                    new_entries = feed_processor.fetch_new_entries(since_date=latest_date)
                    
                    # Filter entries to only include those from this feed
                    for entry in new_entries:
                        if entry.get('_feed_url') == feed_url:
                            entries.append(entry)
                else:
                    # For feeds with no existing documents, fetch all entries
                    entries = feed_processor.fetch_rss_entries(feed_url)
                    # Add source feed information
                    for entry in entries:
                        entry['_feed_url'] = feed_url
                
                if not entries:
                    print(f"\nNo new entries found in RSS feed: {feed_url}")
                    continue
                    
                print(f"\nFound {len(entries)} new entries from {feed_url}. Processing...")
                
                # Process entries and save to respective feed directory
                new_documents = feed_processor.process_entries(entries)
                if new_documents:
                    all_new_documents.extend(new_documents)
                    print(f"\nSuccessfully processed {len(new_documents)} new documents from {feed_url}")
                else:
                    print(f"\nNo new documents were processed from {feed_url}")

            # If no new documents across all feeds, return early
            if not all_new_documents:
                print("\nNo new documents were processed from any feed.")
                return True
                
            print(f"\nTotal new documents across all feeds: {len(all_new_documents)}")

            # Load existing vector store
            index = self.load_vector_store()
            if not index:
                print("\nFailed to load vector store.")
                return False

            # Insert documents into the index
            for doc in all_new_documents:
                # Add document to index
                nodes = index.insert(doc)
                
                # Enhance the newly created nodes
                for node in nodes:
                    if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                        node.metadata['chunk_id'] = f"{node.metadata['file_name']}:{node.start_char_idx}-{node.end_char_idx}"

            # Rebuild metadata index to include new documents
            print("\nUpdating metadata index...")
            self.metadata_repository.build_metadata_index(force_rebuild=True)
            
            print(f"\nVector store successfully updated with {len(all_new_documents)} new documents!")
            return True
            
        except Exception as e:
            print(f"\nError updating vector store: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_document_by_id(self, document_id: str):
        """Retrieve a document by ID (filename)"""
        # Search in all feed directories for the document
        all_document_paths = self._get_all_document_paths()
        
        # Filter paths to find the document with matching ID
        matching_paths = [p for p in all_document_paths if p.name.startswith(f"{document_id}.txt") or p.name.startswith(f"{document_id}_")]
        
        if not matching_paths:
            return None
            
        try:
            file_path = matching_paths[0]
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                
            # Extract metadata from text
            metadata = self._extract_metadata_from_text(text)
            metadata["file_name"] = file_path.name
            
            # Add feed_name based on directory structure
            relative_path = file_path.relative_to(self.cache_dir)
            parts = relative_path.parts
            if len(parts) > 1:  # If file is in a subdirectory
                metadata['feed_name'] = parts[0]  # First part is the feed directory name
            else:
                metadata['feed_name'] = 'unknown'  # For files directly in cache dir
            
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
                    doc = Document(text=doc_text, metadata=metadata)
                    documents_with_scores.append((doc, score))
                    
            return documents_with_scores
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
            
    def get_latest_document_date(self):
        """Get the date of the most recent document in the metadata index"""
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()
            
        if not self.metadata_repository.metadata_list:
            return None
            
        # Metadata list is sorted by date (newest first), so get the first entry
        try:
            latest_entry = self.metadata_repository.metadata_list[0]
            return latest_entry.get('date')
        except (IndexError, KeyError):
            return None
    
    def get_latest_document_dates_by_feed(self):
        """Get the latest document date for each feed source"""
        # Ensure metadata repository is loaded
        if not self.metadata_repository.is_loaded:
            self.metadata_repository.load_metadata_index()
            
        if not self.metadata_repository.metadata_list:
            return {}
            
        latest_dates = {}
        
        # Group metadata entries by feed name
        feed_entries = {}
        for entry in self.metadata_repository.metadata_list:
            feed_name = entry.get('feed_name', 'unknown')
            if feed_name not in feed_entries:
                feed_entries[feed_name] = []
            feed_entries[feed_name].append(entry)
        
        # Get latest date for each feed
        for feed_name, entries in feed_entries.items():
            # Sort entries by date (newest first)
            sorted_entries = sorted(entries, key=lambda x: x.get('date', ''), reverse=True)
            if sorted_entries:
                latest_dates[feed_name] = sorted_entries[0].get('date')
                
        return latest_dates

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
                metadata[key.lower()] = value
                
        return metadata
        
    def _extract_filename_from_node(self, node):
        """Extract filename from node using multiple strategies"""
        # First try chunk_id
        chunk_id = node.metadata.get('chunk_id', '')
        if chunk_id and ':' in chunk_id:
            return chunk_id.split(':', 1)[0]
        
        # Try direct filename
        file_name = node.metadata.get('file_name', '')
        if file_name:
            return file_name
        
        # Try embedded metadata
        if hasattr(node, 'text') and node.text:
            # Try to extract from embedded metadata
            node_metadata = self._extract_embedded_metadata(node.text)
            file_name = node_metadata.get('file_name', '')
            if file_name:
                return file_name
            
            # Try to find it in the text
            import re
            file_match = re.search(r'File: (.*?)\n', node.text)
            if file_match:
                return file_match.group(1).strip()
            
            # Try to match by title if possible
            title_match = re.search(r'Title: (.*?)\n', node.text)
            if title_match and self.metadata_repository.is_loaded:
                title = title_match.group(1).strip()
                # Search metadata repository for this title
                for meta in self.metadata_repository.metadata_list:
                    if meta.get('title', '') == title:
                        return meta.get('file_name', '')
        
        # No filename found
        return ''

    def _extract_embedded_metadata(self, text):
        """Extract metadata embedded in node text"""
        metadata = {}
        if not text:
            return metadata
            
        lines = text.split('\n')
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
                metadata[key.lower()] = value
        
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
            