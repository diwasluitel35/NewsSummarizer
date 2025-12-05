import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

model_path = 'diwasluitel/NewsSummarizer'

def load_model():
    global tokenizer, model
    try:
        print(f"Loading model: {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model (Demo Mode Active): {e}")
        tokenizer = None
        model = None

load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    news = data.get('news', '')

    if not news:
        return jsonify({'error': 'Please enter news to summarize.'}), 400

    if model and tokenizer:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            inputs = tokenizer(news, padding=False, max_length=1024, truncation=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, min_new_tokens=30, num_beams=6, no_repeat_ngram_size=3, early_stopping=False)

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return jsonify({"summary": summary})
            
        except Exception as e:
            return jsonify({'error': f"Summarizaton failed: {str(e)}"}), 500
    else:
        return jsonify({
            "summary": "Error!"
        })

if __name__ == '__main__':
    app.run(debug=True)