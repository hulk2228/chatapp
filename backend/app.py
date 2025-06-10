from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
app = Flask(__name__)
CORS(app)  # Allow React frontend access

client = InferenceClient(token=os.getenv("HF_API_TOKEN"))

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message"}), 400
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": user_message}],
            model="HuggingFaceH4/zephyr-7b-beta",  # Replace with your model
            max_tokens=500
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
