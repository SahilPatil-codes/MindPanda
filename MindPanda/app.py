from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from mental_health_assistant import mental_health_assistant
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})
app.secret_key = os.urandom(24)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def handle_message():
    try:
        user_input = request.json.get('message', '')
        response = mental_health_assistant.get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)