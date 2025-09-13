AI-Powered Semantic Q&A Platform
A simple web application that answers questions using a powerful AI model from Groq. It cleverly saves answers to common questions in a local database to provide instant, cost-free responses for recurring queries.

How It Works
You ask a question on the webpage.

The app checks its local database to see if a similar question has been asked frequently (at least 3 times).
If yes, it gives you the answer instantly from the database.
If no, it sends your question to the Groq AI, gets the answer, shows it to you, and saves it for future use.

##Full Implementation Guide
Step 1: Get the Code
First, clone the repository to your local machine.
git clone [https://github.com/nidheeshkumarn/grok-semantic-qa-platform.git](https://github.com/nidheeshkumarn/grok-semantic-qa-platform.git)
cd grok-semantic-qa-platform

Step 2: Install the Required Libraries
Open your Command Prompt
py -m pip install Flask sentence-transformers torch requests numpy

Step 3: Create Your Groq API Key
The application needs an API key to talk to the AI.
Go to the GroqCloud Console: console.groq.com
Create an API Key and Copy the key and save it. It will only be showed once.

Step 4: Set the API Key on Your Computer
You must store your API key in an environment variable so your code can access it securely.
Open a new Command Prompt.
Run the following command
setx GROK_API_KEY "PASTE_YOUR_NEW_GROQ_API_KEY_HERE" #(Place the key here)

Important: Close this Command Prompt window. The setx command only applies to future windows.

Step 5: (Optional) How to Change the AI Model
If you want to use a different AI model in the future, you only need to edit one line of code.
Open the app.py file in a text editor.
Find the get_grok_answer function.
Change the value of "model" inside the payload dictionary to your desired model name.

# Inside app.py
payload = {
    "model": "openai/gpt-oss-120b", # <-- EDIT THIS LINE
    "messages": [{"role": "user", "content": question}],
    "temperature": 0.7
}

Step 6: Run the Project!
You're all set. Open a new Command Prompt, navigate to your project folder, and run the application.
cd D:\grok_qa_project

# Run the app
python app.py

Your Q&A platform is now running at http://127.0.0.1:5000.
