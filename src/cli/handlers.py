import shutil
import sys
import re

from core.feed_processor import FeedProcessor
from core.vector_store import VectorStoreManager
from core.query_engine import QueryEngine
import config

class MenuHandlers:
    """Handlers for menu options"""
    
    @staticmethod
    def handle_archive_rss():
        """Archive RSS feed entries"""
        feed_processor = FeedProcessor()
        
        cache_dir_has_content = False
        if config.CACHE_DIR.exists():
            # Check if directory has any text files
            cache_dir_has_content = any(config.CACHE_DIR.glob("*.txt"))
        
        if cache_dir_has_content:
            from cli.menu import Menu
            if not Menu.confirm_action("RSS archive already exists. Do you want to update it?"):
                return
                
        print("\nFetching RSS entries...")
        all_entries = feed_processor.fetch_all_feeds()
        
        feed_processor.process_entries(all_entries)
        print("\nRSS feed archived successfully!")
    
    @staticmethod
    def handle_create_vector_store():
        """Create a new vector store"""
        print("DEBUG: Starting create_vector_store handler")
        vector_store_manager = VectorStoreManager()
        
        # First, check if the RSS cache exists and has content
        cache_dir_has_content = False
        if config.CACHE_DIR.exists():
            print(f"DEBUG: Cache dir exists at {config.CACHE_DIR}")
            # Check for files directly in cache dir
            direct_files = list(config.CACHE_DIR.glob("*.txt"))
            print(f"DEBUG: Found {len(direct_files)} files directly in cache dir")
            
            # Check subdirectories
            subdirs = [d for d in config.CACHE_DIR.iterdir() if d.is_dir()]
            print(f"DEBUG: Found {len(subdirs)} subdirectories")
            
            for subdir in subdirs:
                subdir_files = list(subdir.glob("*.txt"))
                print(f"DEBUG: Subdir {subdir.name} has {len(subdir_files)} files")
                if subdir_files:
                    cache_dir_has_content = True
            
            # Original check
            cache_dir_has_content = cache_dir_has_content or any(config.CACHE_DIR.glob("*.txt"))
                
        if not cache_dir_has_content:
            print("\nError: No RSS archive found or archive is empty. Please run option 1 first.")
            return
        
        # Next, check if vector store exists
        vector_store_exists = config.VECTOR_STORE_DIR.exists() and any(config.VECTOR_STORE_DIR.iterdir())
        
        if vector_store_exists:
            from cli.menu import Menu
            if not Menu.confirm_action("Vector store already exists. Do you want to recreate it?"):
                return
        
        print("DEBUG: Calling vector_store_manager.create_vector_store")
        vector_store_manager.create_vector_store(overwrite=True)
    
    @staticmethod
    def handle_load_vector_store():
        """Load the vector store and initialize query engine"""
        query_engine = QueryEngine()
        success = query_engine.initialize()
        return query_engine if success else None
    
    @staticmethod
    def handle_load_chat(query_engine):
        """Start the chat interface with streaming support and date filtering"""
        if not query_engine:
            print("\nError: No query engine loaded. Please run option 3 first.")
            return
                
        print("\nStarting chat interface. Type 'exit' to return to menu.")
        print("For date filtering, use format: --from=YYYY-MM-DD --to=YYYY-MM-DD after your query.")
        
        while True:
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            if query.lower() == 'exit':
                break
                
            # Parse date filters from query if present
            start_date = None
            end_date = None
            
            from_match = re.search(r'--from=(\d{4}-\d{2}-\d{2})', query)
            if from_match:
                start_date = from_match.group(1)
                # Remove filter from query text
                query = re.sub(r'--from=\d{4}-\d{2}-\d{2}', '', query).strip()
                
            to_match = re.search(r'--to=(\d{4}-\d{2}-\d{2})', query)
            if to_match:
                end_date = to_match.group(1)
                # Remove filter from query text
                query = re.sub(r'--to=\d{4}-\d{2}-\d{2}', '', query).strip()
            
            if start_date or end_date:
                print(f"Filtering documents between {start_date or 'earliest'} and {end_date or 'latest'}")
                
            try:
                # The query method now handles streaming responses and prints tokens
                response = query_engine.query(query, start_date=start_date, end_date=end_date)
                print()  # Add a newline after streaming completes
                
                # Format the sources
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    sources_formatted = query_engine.format_sources_only(response.source_nodes)
                    print(sources_formatted)
            except Exception as e:
                print(f"\nError: {e}")
    
    @staticmethod
    def handle_delete_rss_archive():
        """Delete the RSS archive directory"""
        if not config.CACHE_DIR.exists():
            print("\nNo RSS archive found.")
            return
            
        from cli.menu import Menu
        if Menu.confirm_action("Are you sure you want to delete the RSS archive?"):
            try:
                shutil.rmtree(config.CACHE_DIR)
                print("\nRSS archive deleted successfully!")
            except Exception as e:
                print(f"\nError deleting RSS archive: {e}")
    
    @staticmethod
    def handle_delete_vector_store():
        """Delete the vector store directory"""
        if not config.VECTOR_STORE_DIR.exists():
            print("\nNo vector store found.")
            return
            
        from cli.menu import Menu
        if Menu.confirm_action("Are you sure you want to delete the vector store?"):
            try:
                shutil.rmtree(config.VECTOR_STORE_DIR)
                print("\nVector store deleted successfully!")
            except Exception as e:
                print(f"\nError deleting vector store: {e}")
                
    @staticmethod
    def handle_rebuild_metadata_index():
        """Rebuild the metadata index"""
        from core.metadata_repository import MetadataRepository
        import config
        
        metadata_repo = MetadataRepository(config.CACHE_DIR)
        print("\nRebuilding metadata index...")
        success = metadata_repo.build_metadata_index(force_rebuild=True)
        
        if success:
            print(f"\nMetadata index rebuilt successfully with {len(metadata_repo.metadata_list)} entries")
        else:
            print("\nFailed to rebuild metadata index")
            
    @staticmethod
    def handle_update_vector_store():
        """Update the vector store with new RSS feed entries"""
        vector_store_manager = VectorStoreManager()
        success = vector_store_manager.update_vector_store()
        
        if success:
            print("\nVector store updated successfully!")
        else:
            print("\nFailed to update vector store. Check the error messages above.")
    
    @staticmethod
    def handle_configuration():
        """Manage program configuration"""
        from cli.menu import Menu
        
        while True:
            Menu.display_config_menu()
            choice = Menu.get_choice(5)
            
            if choice == 0:
                break
                
            elif choice == 1:
                MenuHandlers._change_embedding_model()
                
            elif choice == 2:
                MenuHandlers._change_chat_model()
                
            elif choice == 3:
                MenuHandlers._change_threads()
                
            elif choice == 4:
                MenuHandlers._change_temperature()
                
            elif choice == 5:
                MenuHandlers._manage_rss_feeds()
    
    @staticmethod
    def _change_embedding_model():
        """Change the embedding model"""
        print("\nAvailable embedding models:")
        print(f"1. BGE-M3 ({config.BGE_M3})")
        print(f"2. GTE-Small ({config.GTE_SMALL})")
        
        from cli.menu import Menu
        model_choice = Menu.get_choice(2)
        
        if model_choice == 1:
            config.CURRENT_EMBED = config.BGE_M3
        elif model_choice == 2:
            config.CURRENT_EMBED = config.GTE_SMALL
            
        print(f"Embedding model set to: {config.CURRENT_EMBED}")
    
    @staticmethod
    def _change_chat_model():
        """Change the chat model"""
        print("\nAvailable chat models:")
        print("1. Qwen 2.5 3B (Smallest)")
        print("2. Qwen 2.5 7B (Medium)")
        print("3. Qwen 2.5 14B (Largest)")
        
        from cli.menu import Menu
        model_choice = Menu.get_choice(3)
        
        if model_choice == 1:
            config.CURRENT_LLM = config.QWEN2_5_3B
        elif model_choice == 2:
            config.CURRENT_LLM = config.QWEN2_5_7B
        elif model_choice == 3:
            config.CURRENT_LLM = config.QWEN2_5_14B
            
        print(f"Chat model set to: {config.CURRENT_LLM}")
    
    @staticmethod
    def _change_threads():
        """Change the number of threads"""
        while True:
            try:
                threads = int(input("\nEnter number of threads (1-16): ").strip())
                if 1 <= threads <= 16:
                    config.NUM_THREADS = threads
                    print(f"Threads set to: {config.NUM_THREADS}")
                    break
                else:
                    print("Invalid number of threads. Must be between 1 and 16.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    @staticmethod
    def _change_temperature():
        """Change the temperature setting"""
        while True:
            try:
                temp = float(input("\nEnter temperature (0.0-1.0): ").strip())
                if 0.0 <= temp <= 1.0:
                    config.TEMPERATURE = temp
                    print(f"Temperature set to: {config.TEMPERATURE}")
                    break
                else:
                    print("Invalid temperature. Must be between 0.0 and 1.0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    @staticmethod
    def _manage_rss_feeds():
        """Add or remove RSS feeds"""
        print("\nCurrent RSS feeds:")
        for i, feed in enumerate(config.RSS_FEED_URLS, 1):
            print(f"{i}. {feed}")
            
        print("\nOptions:")
        print("1. Add feed")
        print("2. Remove feed")
        print("3. Back")
        
        from cli.menu import Menu
        feed_choice = Menu.get_choice(3)
        
        if feed_choice == 1:
            new_feed = input("Enter new RSS feed URL: ").strip()
            if new_feed and new_feed not in config.RSS_FEED_URLS:
                config.RSS_FEED_URLS.append(new_feed)
                print("Feed added successfully!")
        elif feed_choice == 2:
            if len(config.RSS_FEED_URLS) > 1:
                del_idx = int(input("Enter feed number to remove: ").strip()) - 1
                if 0 <= del_idx < len(config.RSS_FEED_URLS):
                    removed = config.RSS_FEED_URLS.pop(del_idx)
                    print(f"Removed feed: {removed}")
                else:
                    print("Invalid feed number.")
            else:
                print("Cannot remove the last RSS feed.")

    @staticmethod
    def handle_test_llm(query_engine):
        """Test if the LLM is functioning properly"""
        if not query_engine:
            print("\nError: No query engine loaded. Please run option 3 first.")
            return
        
        print("\nTesting LLM functionality...")
        result = query_engine.test_llm()
        print(result)
