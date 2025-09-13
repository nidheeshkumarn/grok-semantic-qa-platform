import os
import requests
import json

# --- Configuration ---
API_KEY = os.environ.get("GROK_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Check if the API key is available
if not API_KEY:
    print("FATAL ERROR: The GROK_API_KEY environment variable is not set.")
    print("Please close this terminal, open a new one, and try again.")
    exit()

print(f"API Key found. Starting with key: ...{API_KEY[-4:]}")

# --- API Request Details ---
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gemma-7b-it", # A reliable and standard model
    "messages": [{"role": "user", "content": "Explain the importance of low-latency LLMs"}],
}

# --- Make the API Call ---
print("\nSending request to Groq API...")
try:
    response = requests.post(API_URL, headers=headers, json=payload)

    # This function will raise an error for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # If the request was successful
    print("\n✅ SUCCESS! Groq API responded correctly.")
    data = response.json()
    print("\nResponse from Groq:")
    print(json.dumps(data, indent=2))

except requests.exceptions.HTTPError as http_err:
    print(f"\n❌ HTTP ERROR occurred: {http_err}")
    print(f"Status Code: {response.status_code}")
    # THIS IS THE MOST IMPORTANT PART: Print the detailed error from Groq
    print("\n--- DETAILED ERROR RESPONSE ---")
    try:
        print(response.json())
    except json.JSONDecodeError:
        print(response.text)
    print("-----------------------------")

except Exception as err:
    print(f"\n❌ An OTHER ERROR occurred: {err}")
