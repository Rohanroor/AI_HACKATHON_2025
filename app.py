from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify API key is loaded
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize Flask app and configure Gemini API
app = Flask(__name__)
genai.configure(api_key=api_key)
generation_config = genai.GenerationConfig(
    temperature=0.9,
    candidate_count=1,
)
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
model = genai.GenerativeModel(
    model_name='gemini-pro',
    generation_config=generation_config,
    safety_settings=safety_settings
)
# Initialize chat with doctor context
doctor_context = """You are a knowledgeable and empathetic medical doctor with expertise in mental health. 
Your role is to provide supportive, professional medical guidance while maintaining a compassionate approach. 
While you can offer general medical advice and information, always remind users that you cannot replace their actual healthcare provider 
and encourage them to seek professional medical help for specific diagnoses or treatment plans."""

chat = model.start_chat(history=[
    {'role': 'user', 'parts': ['Please act as a medical doctor with the following context: ' + doctor_context]},
    {'role': 'model', 'parts': ['I understand. I will act as a medical doctor with the specified context and guidelines.']}
])

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
    return render_template('index.html')

@app.route('/mhchatbot')
def chatbot():
    return render_template('mhchatbot.html')

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
