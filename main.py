from chatbot import process_query

def chat_Loop():
    print("Type your queries or 'quit' to exit.")
    messages = []
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            messages = process_query(query, messages)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_Loop()
