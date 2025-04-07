from platform import java_ver
import shutil
from pathlib import Path
from datetime import datetime
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
        
        # Use absolute paths to avoid resolution issues
        cache_dir = self.cache_dir.absolute()
        print(f"DEBUG: Using absolute cache path: {cache_dir}")
        
        # First check if there are any documents directly in the cache directory (old structure)
        direct_files = list(cache_dir.glob("*.txt"))
        print(f"DEBUG: Found {len(direct_files)} direct files")
        if direct_files:
            all_paths.extend(direct_files)
            
        # Then check all subdirectories (new structure)
        subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
        print(f"DEBUG: Found {len(subdirs)} subdirectories")
        
        for subdir in subdirs:
            print(f"DEBUG: Checking subdir: {subdir}")
            subdir_files = list(subdir.glob("*.txt"))
            print(f"DEBUG: Found {len(subdir_files)} files in {subdir.name}")
            all_paths.extend(subdir_files)
        
        print(f"DEBUG: Total document paths found: {len(all_paths)}")    
        return all_paths
    
    def create_vector_store(self, overwrite=False):
        """Create a new vector store from cached documents with improved metadata preservation"""
        all_document_paths = self._get_all_document_paths()
        
        # Add debugging
        print(f"DEBUG: Cache directory: {self.cache_dir} (exists: {self.cache_dir.exists()})")
        print(f"DEBUG: Found {len(all_document_paths)} document paths")
        if len(all_document_paths) > 0:
            print(f"DEBUG: First few paths: {[str(p) for p in all_document_paths[:3]]}")
        
        # Print subdirectory contents
        subdirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        print(f"DEBUG: Found {len(subdirs)} subdirectories in {self.cache_dir}")
        for subdir in subdirs:
            files = list(subdir.glob("*.txt"))
            print(f"DEBUG: Subdir {subdir.name} contains {len(files)} .txt files")
        
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
                    
                    standardized_metadata = {
                        'title': metadata.get('title', 'Untitled'),
                        'date': metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'author': metadata.get('author', 'Unknown Author'),
                        'url': metadata.get('url', 'No URL'),
                        'feed_name': metadata.get('feed_name', 'unknown'),
                        'file_name': metadata.get('file_name', ''),
                        'categories': ','.join(metadata.get('categories', [])),
                        # Add additional fields for filtering
                        'year': metadata.get('date', '').split('-')[0] if metadata.get('date') else '',
                        'month': metadata.get('date', '').split('-')[1] if metadata.get('date') else '',
                        'has_categories': len(metadata.get('categories', [])) > 0
                    }

                    all_documents.append(Document(text=enhanced_text, metadata=standardized_metadata))
                    
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

            embed_model = LLMManager.initialize_embedding_model()
            Settings.embed_model = embed_model

            try:
                llm = LLMManager.initialize_llm()
                Settings.llm = llm
            except Exception as e:
                # Handle LLM initialization error but continue loading vector store
                print(f"Warning: Failed to initialize LLM, will continue with vector store: {e}")
            
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
            # Load metadata repository to check for existing documents
            if not self.metadata_repository.is_loaded:
                success = self.metadata_repository.load_metadata_index()
                if not success:
                    print("\nWarning: Failed to load metadata index. Attempting to rebuild...")
                    success = self.metadata_repository.build_metadata_index(force_rebuild=True)
                    if not success:
                        print("\nError: Failed to build metadata index. Cannot determine latest documents.")
                        return False
            
            # Get the latest document date for each feed source
            latest_dates = self.get_latest_document_dates_by_feed()
            if not latest_dates:
                print("\nNo existing documents found in metadata. You should create the vector store instead.")
                return False
                
            # Create a feed processor for fetching new entries
            feed_processor = FeedProcessor()
            
            # Process each feed configuration individually
            all_new_documents = []
            feed_configs = config.RSS_FEED_CONFIG
            
            # Debug output
            print(f"\nLatest document dates by feed: {latest_dates}")
            print(f"\nProcessing {len(feed_configs)} feed configurations")
            
            for feed_config in feed_configs:
                feed_url = feed_config["url"]
                
                # Get the feed directory name using the same method as in feed_processor
                feed_dir_name = feed_processor._get_feed_directory_name(feed_url)
                latest_date = latest_dates.get(feed_dir_name, None)
                
                print(f"\nChecking feed: {feed_url}")
                print(f"Feed directory name: {feed_dir_name}")
                print(f"Latest document date: {latest_date or 'None'}")
                
                # Fetch new entries since last document date
                if latest_date:
                    print(f"Fetching entries newer than {latest_date}...")
                    
                    # Specific feed processor for this URL
                    single_feed_processor = FeedProcessor([feed_url])
                    
                    # Fetch only new entries for this feed
                    new_entries = single_feed_processor.fetch_new_entries(since_date=latest_date)
                    
                    # Mark entries with feed URL for tracking
                    for entry in new_entries:
                        entry['_feed_url'] = feed_url
                        
                    print(f"Found {len(new_entries)} new entries")
                    
                    # Process and save new entries
                    if new_entries:
                        new_documents = single_feed_processor.process_entries(new_entries)
                        if new_documents:
                            all_new_documents.extend(new_documents)
                            print(f"Successfully processed {len(new_documents)} new documents")
                        else:
                            print("No valid documents created from new entries")
                else:
                    print(f"No existing documents for this feed. Fetching all entries...")
                    
                    # Specific feed processor for this URL
                    single_feed_processor = FeedProcessor([feed_url])
                    
                    # Fetch all entries for this feed
                    entries = single_feed_processor.fetch_rss_entries(feed_url)
                    
                    # Mark entries with feed URL for tracking
                    for entry in entries:
                        entry['_feed_url'] = feed_url
                        
                    print(f"Found {len(entries)} entries")
                    
                    # Process and save entries
                    if entries:
                        new_documents = single_feed_processor.process_entries(entries)
                        if new_documents:
                            all_new_documents.extend(new_documents)
                            print(f"Successfully processed {len(new_documents)} new documents")
                        else:
                            print("No valid documents created from entries")
            
            # If no new documents across all feeds, return early
            if not all_new_documents:
                print("\nNo new documents were processed from any feed.")
                return True
                
            print(f"\nTotal new documents across all feeds: {len(all_new_documents)}")

            # Load existing vector store
            print("\nLoading existing vector store...")
            index = self.load_vector_store()
            if not index:
                print("Failed to load vector store.")
                return False

            # Initialize embedding model if needed for document insertion
            if not hasattr(Settings, 'embed_model') or Settings.embed_model is None:
                from core.llm_manager import LLMManager
                print("Initializing embedding model...")
                embed_model = LLMManager.initialize_embedding_model()
                Settings.embed_model = embed_model
                
            # Configure node parser if not already set
            if not hasattr(Settings, 'node_parser') or Settings.node_parser is None:
                from llama_index.core.node_parser import SentenceSplitter
                print("Initializing node parser...")
                Settings.node_parser = SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    paragraph_separator="\n\n"
                )

            # Insert documents into the index
            print(f"Inserting {len(all_new_documents)} new documents into the vector store...")
            successful_inserts = 0
            
            # Create a sentence splitter for document parsing
            from llama_index.core.node_parser import SentenceSplitter
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            
            for i, doc in enumerate(all_new_documents):
                try:
                    # Get document name for logging
                    doc_name = "unknown"
                    if hasattr(doc, 'metadata'):
                        # Try different metadata fields to identify the document
                        if 'file_name' in doc.metadata:
                            doc_name = doc.metadata['file_name']
                        elif 'title' in doc.metadata:
                            doc_name = doc.metadata['title']
                    
                    print(f"Processing document {i+1}/{len(all_new_documents)}: {doc_name}")
                    
                    # Ensure document metadata is compatible with ChromaDB requirements
                    if hasattr(doc, 'metadata'):
                        # Process categories if present to ensure it's a string
                        if 'categories' in doc.metadata and isinstance(doc.metadata['categories'], (list, tuple)):
                            doc.metadata['categories'] = ','.join(str(cat) for cat in doc.metadata['categories'] if cat)
                        
                        # Ensure all metadata values are of compatible types
                        for key, value in list(doc.metadata.items()):
                            if not isinstance(value, (str, int, float, type(None))):
                                doc.metadata[key] = str(value)
                    
                    # Use the alternative method directly (parse document into nodes first, then insert)
                    nodes = parser.get_nodes_from_documents([doc])
                    
                    if nodes:
                        # Insert individual nodes
                        for node in nodes:
                            # Copy document metadata to node
                            for key, value in doc.metadata.items():
                                node.metadata[key] = value
                                
                            # Add chunk_id to metadata for tracking
                            if 'file_name' in node.metadata:
                                node.metadata['chunk_id'] = f"{node.metadata['file_name']}:{node.start_char_idx}-{node.end_char_idx}"
                            
                            # Insert node directly
                            index.docstore.add_documents([node])
                        
                        successful_inserts += 1
                        print(f"Successfully inserted document: {doc_name}")
                    else:
                        print(f"Warning: Failed to create nodes for document: {doc_name}")
                        continue
                    
                    if (i + 1) % 5 == 0 or i + 1 == len(all_new_documents):
                        print(f"Progress: {i + 1}/{len(all_new_documents)} documents processed (successful: {successful_inserts})")
                        
                except Exception as e:
                    print(f"Error inserting document {i+1} ({doc_name}): {e}")
                    print(f"Document details: id={getattr(doc, 'doc_id', 'unknown')}, "
                        f"length={len(doc.text) if hasattr(doc, 'text') else 'unknown'}")
                    print(f"Metadata keys: {list(doc.metadata.keys()) if hasattr(doc, 'metadata') else 'none'}")
                    continue

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
