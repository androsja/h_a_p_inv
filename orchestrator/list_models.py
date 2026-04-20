import os
from google import genai

def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No API Key found")
        return
    
    client = genai.Client(api_key=api_key)
    print("Available models:")
    try:
        for model in client.models.list():
            print(f"- {model.name} (DisplayName: {model.display_name})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
