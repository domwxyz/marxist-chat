import signal
import sys

from cli.menu import Menu
from cli.handlers import MenuHandlers

def signal_handler(sig, frame):
    """Handle CTRL + C exit signal"""
    Menu.exit_program()

def main():
    """Main application entry point"""
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize query engine variable
    query_engine = None
    
    # Main application loop
    while True:
        try:
            Menu.display_main_menu()
            choice = Menu.get_choice(8)  # Update this from 7 to 8
            
            if choice == 0:
                Menu.exit_program()
                
            elif choice == 1:
                MenuHandlers.handle_archive_rss()
                
            elif choice == 2:
                MenuHandlers.handle_create_vector_store()
                
            elif choice == 3:
                query_engine = MenuHandlers.handle_load_vector_store()
                
            elif choice == 4:
                MenuHandlers.handle_load_chat(query_engine)
                
            elif choice == 5:
                MenuHandlers.handle_delete_rss_archive()
                
            elif choice == 6:
                MenuHandlers.handle_delete_vector_store()
                query_engine = None  # Reset query engine if vector store is deleted
                
            elif choice == 7:
                MenuHandlers.handle_configuration()
                
            elif choice == 8:
                MenuHandlers.handle_test_llm(query_engine)
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
