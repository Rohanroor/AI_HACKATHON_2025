from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pickle
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
    temperature=0.8,
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
doctor_context = """Your name is Rakshak, and your a personal health care companion, trained by team Old Monk. You are specially designed and catered for only one purpose which is to provide mental and physical health care and support to the users of Rakshak health care support platform.  You are a knowledgeable and empathetic medical AI chat bot with expertise in both mental and physical health care. 
Your role is to provide supportive, professional medical guidance while maintaining a compassionate approach. 
While you can offer general medical advice and information, always remind users that you cannot replace their actual healthcare provider 
and encourage them to seek professional medical help for specific diagnoses or treatment plans.
please only stick to your domain while answering, which is mental and physical healthcare sector. And if some one tries to go beyond this domain, just kindly say no to them and ask them to stick to your domain only while mentioning your domains
when someone ask you about yourself no need to give a detailed answer always just give them a presized and short answer."""

chat = model.start_chat(history=[
    {'role': 'user', 'parts': ['Please act as a medical doctor with the following context: ' + doctor_context]},
    {'role': 'model', 'parts': ['I understand. I will act as a medical doctor with the specified context and guidelines.']}
])

def clean_response_text(text):
    # Remove markdown bold syntax
    cleaned = text.replace('**', '')
    
    # Convert numbered lists and bullet points
    cleaned = cleaned.replace('1.', '\n1.') \
                    .replace('2.', '\n2.') \
                    .replace('3.', '\n3.') \
                    .replace('• ', '\n• ')
    
    # Add line breaks before headings
    cleaned = cleaned.replace('Possible Health Conditions:', '\nPossible Health Conditions:\n')
    
    # Preserve existing newlines and split into paragraphs
    paragraphs = [p.strip() for p in cleaned.split('\n') if p.strip()]
    
    # Join paragraphs with double newlines
    cleaned = '\n\n'.join(paragraphs)
    
    # Convert to HTML line breaks (will be used in template)
    return cleaned.replace('\n\n', '<br><br>').replace('\n', '<br>')

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


@app.route('/diabetes', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get JSON data from the request
            data = request.get_json()

            # Convert values
            pregnancies = int(data['pregnancies'])
            glucose = int(data['glucose'])
            blood_pressure = int(data['blood-pressure'])
            skin_thickness = int(data['skin-thickness'])
            insulin = int(data['insulin'])
            bmi = float(data['bmi'])
            diabetes_pedigree = float(data['diabetes-pedigree'])
            age = int(data['age'])

            # Create feature array
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, diabetes_pedigree, age]

            # Load model and predict
            with open('model.pickle', 'rb') as file:
                model = pickle.load(file)
            prediction = model.predict([input_data])[0]

            # Create response message
            if prediction == 0:
                message = "Based on our analysis, you show no signs of diabetes. However, maintain regular check-ups."
            else:
                message = "Our prediction indicates a risk of diabetes. Please consult a healthcare professional."

            return jsonify({'message': message})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # GET request - show form
    return render_template('diabetes.html')

@app.route('/search-facilities')
def search_facilities():
    return render_template('search_facilities.html')

@app.route('/get-nearby-facilities', methods=['POST'])
def get_nearby_facilities():
    try:
        # Real facilities in Sambalpur
        sample_facilities = [
            {
                "name": "Veer Surendra Sai Institute of Medical Sciences and Research (VIMSAR)",
                "type": "Hospital",
                "address": "Burla, Sambalpur, Odisha 768017",
                "distance": "0.5 km",
                "rating": 4.1
            },
            {
                "name": "District Headquarters Hospital",
                "type": "Hospital",
                "address": "Church Rd, Naya Para, Sambalpur, Odisha 768001",
                "distance": "1.2 km",
                "rating": 3.9
            },
            {
                "name": "Sambalpur Diagnostic Centre",
                "type": "Pathlab",
                "address": "Budharaja, Sambalpur, Odisha 768004",
                "distance": "2.1 km",
                "rating": 4.3
            },
            {
                "name": "Lifeline Diagnostics & Research Centre",
                "type": "Pathlab",
                "address": "Near SBI Bank, Budharaja, Sambalpur, Odisha 768004",
                "distance": "2.3 km",
                "rating": 4.0
            },
            {
                "name": "Aditya Care Hospital",
                "type": "Hospital",
                "address": "NH 6, Modipara, Sambalpur, Odisha 768002",
                "distance": "3.0 km",
                "rating": 4.2
            },
            {
                "name": "Sambalpur Cancer Care Hospital",
                "type": "Hospital",
                "address": "Near Ainthapali Chowk, Sambalpur, Odisha 768004",
                "distance": "3.5 km",
                "rating": 4.4
            },
            {
                "name": "City Care Hospital",
                "type": "Hospital",
                "address": "Budharaja, Sambalpur, Odisha 768004",
                "distance": "2.8 km",
                "rating": 4.0
            },
            {
                "name": "Hitech Diagnostic Centre",
                "type": "Pathlab",
                "address": "Modipara, Sambalpur, Odisha 768002",
                "distance": "2.5 km",
                "rating": 4.2
            },
            {
                "name": "Apex Diagnostics",
                "type": "Pathlab",
                "address": "Near Golbazar, Sambalpur, Odisha 768001",
                "distance": "1.8 km",
                "rating": 4.1
            },
            {
                "name": "Ayush Hospital",
                "type": "Hospital",
                "address": "Dhanupali, Sambalpur, Odisha 768001",
                "distance": "2.0 km",
                "rating": 3.8
            },
            {
                "name": "Kalinga Laboratory",
                "type": "Pathlab",
                "address": "Near Court Complex, Sambalpur, Odisha 768001",
                "distance": "1.5 km",
                "rating": 4.0
            },
            {
                "name": "Sun Hospital",
                "type": "Hospital",
                "address": "Budharaja, Sambalpur, Odisha 768004",
                "distance": "2.7 km",
                "rating": 3.9
            }
        ]
        return jsonify({"facilities": sample_facilities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms-diagnosis', methods=['GET', 'POST'])
def symptoms_diagnosis():
    diagnosis = None
    if request.method == 'POST':
        try:
            symptoms = request.form['symptoms']
            
            # Create medical prompt
            prompt = f"""Analyze these symptoms: {symptoms}
            Provide:
            1. Top 3 possible medical conditions (order by probability)
            2. Brief explanation for each
            3. Recommended next steps
            4. Urgency level (emergency/urgent/routine)
            
            Format clearly and concisely. Use plain text only, no markdown.
            Remind to consult a real doctor."""
            
            # Get Gemini response
            response = model.generate_content(prompt)
            
            # Clean and format response
            cleaned = clean_response_text(response.text)
            
            # Ensure line breaks are preserved in HTML
            diagnosis = cleaned.replace('\n', '<br>')
            
        except Exception as e:
            return render_template('symptoms.html', 
                diagnosis=f"Error analyzing symptoms: {str(e)}")

    return render_template('symptoms.html', diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
