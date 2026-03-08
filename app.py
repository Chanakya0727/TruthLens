import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import re
import string

# Point Flask to the frontend folder to serve static files (HTML, CSS, JS)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
# Enable CORS so our frontend can easily connect
CORS(app)

# Define paths
MODEL_DIR = r"c:\Users\chana\OneDrive\Desktop\project\ai_fake_news\model"

print("Loading model and vectorizer...")
try:
    model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), 'rb'))
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), 'rb'))
    print("Success: Model loaded!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Will gracefully fail when endpoints are hit if model isn't finished yet.

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        news_text = data['text']
        
        if len(news_text.strip()) < 10:
             return jsonify({"error": "Text is too short to analyze."}), 400

        # Preprocess text
        clean_text = wordopt(news_text)

        # Vectorize
        transformed_text = vectorizer.transform([clean_text])

        # Predict
        prediction = model.predict(transformed_text)
        
        # Calculate probabilities to show confidence
        probabilities = model.predict_proba(transformed_text)[0]
        confidence_fake = probabilities[0] * 100
        confidence_real = probabilities[1] * 100

        # Mapping: 0 = Fake, 1 = Real
        if prediction[0] == 1:
            result_label = "Real News"
            confidence = round(confidence_real, 2)
        else:
            result_label = "Fake News"
            confidence = round(confidence_fake, 2)

        return jsonify({
            "prediction": result_label,
            "confidence": f"{confidence}%",
            "is_fake": bool(prediction[0] == 0)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Healthy API"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
