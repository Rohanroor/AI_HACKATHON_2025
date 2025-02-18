from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API key is loaded
api_key = os.getenv('GOOGLE_API_KEY')
print(f"Loaded API key: {api_key[:10]}...") # This will show just the first 10 characters
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize Flask app and configure Gemini API
app = Flask(__name__)
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])  # Initialize a persistent chat session

def clean_response_text(text):
    # Remove markdown bold syntax
    cleaned = text.replace('**', '')
    
    # Add proper spacing after colons
    cleaned = cleaned.replace(':', ': ')
    
    # Add line breaks before bullet points
    cleaned = cleaned.replace('* ', '\n• ')
    
    # Add line breaks between paragraphs (when there's a period followed by space)
    cleaned = cleaned.replace('. ', '.\n\n')
    
    # Remove any extra blank lines (more than 2)
    cleaned = '\n'.join([line for line in cleaned.split('\n') if line.strip()])
    
    # Normalize multiple spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

@app.route('/')
def home():
    return render_template('mschatbot.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        user_message = request.json['message']
        
        # Generate response using Gemini with chat history
        response = chat.send_message(user_message)
        
        # Clean the response text before sending
        cleaned_response = clean_response_text(response.text)
        
        return jsonify({
            'response': cleaned_response
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
