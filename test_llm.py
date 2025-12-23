from llama_index.llms.ollama import Ollama

def test_connection():
    print("Initializing Ollama with Llama 3.1...")
    
    # Connect to the local Ollama instance
    # request_timeout is increased because local models can take a moment to load into RAM
    llm = Ollama(model="llama3.1", request_timeout=60.0)

    print("Asking a test question...")
    response = llm.complete("Explain clearly what is a derivative in finance in one sentence.")

    print("\n--- Response from Chatbot ---")
    print(response)

if __name__ == "__main__":
    test_connection()