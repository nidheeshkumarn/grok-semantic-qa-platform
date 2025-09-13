import sqlite3
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import requests
import os
import numpy as np # <-- 1. ADDED numpy import for handling embeddings

# --- Configuration ---
app = Flask(__name__)
DB_FILE = 'qa_database.db'
SIMILARITY_THRESHOLD = 0.95 # Adjust this value (0.0 to 1.0) based on desired strictness
FREQUENCY_THRESHOLD = 3
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- 2. REMOVED Hardcoded API Key ---
# The key MUST be loaded from an environment variable for security.
GROK_API_KEY = os.environ.get("GROK_API_KEY")

# Load a pre-trained model for calculating sentence similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL UNIQUE,
            answer TEXT NOT NULL,
            frequency INTEGER NOT NULL DEFAULT 1,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- Helper Functions ---
def get_grok_answer(question):
    """Fetches an answer from the Groq API."""
    if not GROK_API_KEY:
        print("ERROR: GROK_API_KEY environment variable not set.")
        return "Sorry, the application is not configured with an API key."

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-120b", # Using the correct, current model
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.7
    }
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, there was an error communicating with the AI service."

# --- 3. OPTIMIZED this function ---
def find_similar_question(user_question_embedding):
    """Finds a semantically similar question in the database EFFICIENTLY."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer, frequency, embedding FROM questions")
    all_questions = cursor.fetchall()
    conn.close()

    if not all_questions:
        return None

    # Instead of re-encoding, load the stored embeddings and compare.
    for q_id, db_question, answer, freq, db_embedding_blob in all_questions:
        # Convert the BLOB from the DB back into a numpy array
        db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float32)
        
        # Reshape if necessary, depending on the model's output shape
        # For 'all-MiniLM-L6-v2', the output dimension is 384
        if db_embedding.shape[0] != 384:
            # This is a fallback; usually the shape is correct.
             db_embedding = db_embedding.reshape(1, -1)

        similarity = util.cos_sim(user_question_embedding, db_embedding)
        
        if similarity.item() > SIMILARITY_THRESHOLD:
            return {'id': q_id, 'question': db_question, 'answer': answer, 'frequency': freq}
            
    return None

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    user_question_embedding = model.encode(user_question, convert_to_tensor=False)

    similar_q = find_similar_question(user_question_embedding)

    if similar_q and similar_q['frequency'] >= FREQUENCY_THRESHOLD:
        print(f"Found frequent similar question in DB: '{similar_q['question']}'")
        return jsonify({'answer': similar_q['answer'], 'source': 'Database'})

    print("No frequent match in DB. Querying Grok API...")
    answer = get_grok_answer(user_question)

    # Do not store questions if the API call failed
    if "Sorry," in answer:
         return jsonify({'answer': answer, 'source': 'Error'})

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if similar_q:
        new_freq = similar_q['frequency'] + 1
        cursor.execute("UPDATE questions SET frequency = ? WHERE id = ?", (new_freq, similar_q['id']))
        print(f"Updated frequency for question ID {similar_q['id']} to {new_freq}")
    else:
        # Convert numpy array to bytes (BLOB) for storage
        embedding_blob = user_question_embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO questions (question, answer, frequency, embedding) VALUES (?, ?, ?, ?)",
            (user_question, answer, 1, embedding_blob)
        )
        print(f"Added new question to DB: '{user_question}'")
    
    conn.commit()
    conn.close()

    return jsonify({'answer': answer, 'source': 'Grok API'})

# --- Main Execution ---
if __name__ == '__main__':
    # Add this check to ensure numpy is installed
    try:
        import numpy
    except ImportError:
        print("Error: numpy is not installed. Please run 'pip install numpy'")
        exit()

    init_db()
    app.run(debug=True)
