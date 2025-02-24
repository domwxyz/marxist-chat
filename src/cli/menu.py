import sys

class Menu:
    """Menu system for the CLI interface"""
    
    @staticmethod
    def display_main_menu():
        """Display the main menu options"""
        print("\nRSS RAG Bot Menu")
        print("----------------")
        print("1. Archive RSS Feed")
        print("2. Create Vector Store")
        print("3. Load Vector Store")
        print("4. Load Chat")
        print("5. Delete RSS Archive")
        print("6. Delete Vector Store")
        print("7. Configuration")
        print("0. Exit")
        print("\nNote: For first time setup, run options 1, 2, 3, and 4 in order.")
    
    @staticmethod
    def display_config_menu():
        """Display configuration options"""
        print("\nConfiguration Menu")
        print("-----------------")
        print("1. Change Embedding Model")
        print("2. Change Chat Model")
        print("3. Change Number of Threads")
        print("4. Change Temperature")
        print("5. Add/Remove RSS Feeds")
        print("0. Back to Main Menu")
    
    @staticmethod
    def get_choice(max_choice):
        """Get a valid menu choice from the user"""
        while True:
            try:
                choice = input(f"\nEnter your choice (0-{max_choice}): ").strip()
                choice_num = int(choice)
                if 0 <= choice_num <= max_choice:
                    return choice_num
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {max_choice}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    @staticmethod
    def confirm_action(prompt):
        """Confirm an action with the user"""
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        return response == 'y'
    
    @staticmethod
    def exit_program():
        """Exit the program gracefully"""
        print("\nGoodbye!")
        sys.exit(0)
