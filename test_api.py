from langsmith import Client

try:
    client = Client()
    print("LangSmith connected successfully!")
    print(f"API URL: {client.api_url}")
except Exception as e:
    print(f"Connection failed: {e}")
