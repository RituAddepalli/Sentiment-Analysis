# for render 

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import nltk
import os

app = Flask(__name__, static_folder="static")  # Set static folder to where Angular build is
CORS(app)

# Auto-download required NLTK resources
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

sia = SentimentIntensityAnalyzer()

# Emoji mapping
emoji_map = {
    "Positive": "😊",
    "Negative": "😞",
    "Neutral": "😐",
    "Mixed": "😶"
}

# Common sentiment words for highlighting
possible_sentiment_words = [
    "like", "love", "hate", "dislike", "happy", "sad", "angry", "amazing",
    "terrible", "good", "bad", "excellent", "poor", "awesome", "worst", "best",
    "expensive", "cheap", "joy", "disgust", "fun", "boring", "irritating"
]

# Detect tone based on compound score
def detect_tone(compound):
    if compound >= 0.5:
        return "Joy"
    elif 0.05 <= compound < 0.5:
        return "Happy"
    elif -0.5 < compound < 0:
        return "Sad"
    elif compound <= -0.5:
        return "Anger"
    else:
        return "Neutral"

# Negation-aware word highlighting
def get_word_sentiments(text):
    words = word_tokenize(text)
    positive_words = []
    negative_words = []

    for i, w in enumerate(words):
        phrase = " ".join(words[max(0, i-1):i+1])  # previous + current word
        score = sia.polarity_scores(phrase)['compound']

        if score > 0 and w.lower() in possible_sentiment_words:
            positive_words.append(w)
        elif score < 0 and w.lower() in possible_sentiment_words:
            negative_words.append(w)

    return list(set(positive_words)), list(set(negative_words))

# API route
@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    scores = sia.polarity_scores(text)
    compound = scores['compound']

    positive_words, negative_words = get_word_sentiments(text)

    if positive_words and negative_words:
        sentiment = "Mixed"
    elif compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = round(abs(compound) * 100, 2)
    tone = detect_tone(compound)

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "polarity": round(compound, 3),
        "confidence": f"{confidence}%",
        "emoji": emoji_map[sentiment],
        "positive_words": positive_words,
        "negative_words": negative_words,
        "tone": tone
    })

# --- Serve Angular frontend ---
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
























# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# import nltk

# app = Flask(__name__)
# CORS(app)

# # Auto-download required NLTK resources
# try:
#     nltk.data.find("sentiment/vader_lexicon.zip")
# except LookupError:
#     nltk.download("vader_lexicon")

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# sia = SentimentIntensityAnalyzer()

# # Emoji mapping
# emoji_map = {
#     "Positive": "😊",
#     "Negative": "😞",
#     "Neutral": "😐",
#     "Mixed": "😶"
# }

# # Common sentiment words for highlighting
# possible_sentiment_words = [
#     "like", "love", "hate", "dislike", "happy", "sad", "angry", "amazing",
#     "terrible", "good", "bad", "excellent", "poor", "awesome", "worst", "best",
#     "expensive", "cheap", "joy", "disgust", "fun", "boring", "irritating"
# ]

# # Detect tone based on compound score
# def detect_tone(compound):
#     if compound >= 0.5:
#         return "Joy"
#     elif 0.05 <= compound < 0.5:
#         return "Happy"
#     elif -0.5 < compound < 0:
#         return "Sad"
#     elif compound <= -0.5:
#         return "Anger"
#     else:
#         return "Neutral"

# # --- Updated negation-aware word highlighting ---
# def get_word_sentiments(text):
#     words = word_tokenize(text)
#     positive_words = []
#     negative_words = []

#     for i, w in enumerate(words):
#         # Check the word along with its preceding word for negation
#         phrase = " ".join(words[max(0, i-1):i+1])  # previous + current word
#         score = sia.polarity_scores(phrase)['compound']

#         # Append to correct list only if it's in possible_sentiment_words
#         if score > 0 and w.lower() in possible_sentiment_words:
#             positive_words.append(w)
#         elif score < 0 and w.lower() in possible_sentiment_words:
#             negative_words.append(w)

#     return list(set(positive_words)), list(set(negative_words))

# # Flask route to analyze sentiment
# @app.route("/analyze", methods=["POST"])
# def analyze_sentiment():
#     data = request.json
#     text = data.get("text", "").strip()

#     if not text:
#         return jsonify({"error": "Text is required"}), 400

#     scores = sia.polarity_scores(text)
#     compound = scores['compound']

#     positive_words, negative_words = get_word_sentiments(text)

#     # Determine sentiment including Mixed
#     if positive_words and negative_words:
#         sentiment = "Mixed"
#     elif compound >= 0.05:
#         sentiment = "Positive"
#     elif compound <= -0.05:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     confidence = round(abs(compound) * 100, 2)
#     tone = detect_tone(compound)

#     return jsonify({
#         "text": text,
#         "sentiment": sentiment,
#         "polarity": round(compound, 3),
#         "confidence": f"{confidence}%",
#         "emoji": emoji_map[sentiment],
#         "positive_words": positive_words,
#         "negative_words": negative_words,
#         "tone": tone
#     })

# @app.route("/")
# def home():
#     return jsonify({"status": "Sentiment Analysis API running"}), 200

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
















# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from nltk.sentiment import SentimentIntensityAnalyzer
# import nltk
# import os

# app = Flask(__name__)
# CORS(app)

# # Auto-download VADER lexicon if not present
# try:
#     nltk.data.find("sentiment/vader_lexicon.zip")
# except LookupError:
#     nltk.download("vader_lexicon")

# sia = SentimentIntensityAnalyzer()

# # Emoji mapping
# emoji_map = {
#     "Positive": "😊",
#     "Negative": "😞",
#     "Neutral": "😐",
#     "Mixed": "😶"
# } 
# # def detect_tone(compound):
# #         if compound >= 0.5:
# #             return "Joy"
# #         elif 0 < compound < 0.5:
# #             return "Happy"
# #         elif -0.5 < compound < 0:
# #             return "Sad"
# #         elif compound <= -0.5:
# #             return "Anger"
# #         else:
# #             return "Neutral"

# def detect_tone(compound):
#     if compound >= 0.05:
#         return "Joy"
#     elif -0.5<compound<0:
#         return "Sad"
#     elif compound <= -0.05:
#         return "Anger"
#     else:
#         return "Neutral"

# def get_word_sentiments(text):
#     words = text.split()
#     positive_words = []
#     negative_words = []

#     for w in words:
#         score = sia.polarity_scores(w)['compound']
#         if score > 0:
#             positive_words.append(w)
#         elif score < 0:
#             negative_words.append(w)

#     return list(set(positive_words)), list(set(negative_words))

# @app.route("/analyze", methods=["POST"])
# def analyze_sentiment():
#     data = request.json
#     text = data.get("text", "").strip()

#     if not text:
#         return jsonify({"error": "Text is required"}), 400

#     scores = sia.polarity_scores(text)
#     compound = scores['compound']

#     positive_words, negative_words = get_word_sentiments(text)

#     # Determine sentiment with Mixed detection
#     if positive_words and negative_words:
#         sentiment = "Mixed"
#     elif compound >= 0.05:
#         sentiment = "Positive"
#     elif compound <= -0.05:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     confidence = round(abs(compound) * 100, 2)
#     tone = detect_tone(compound)

#     return jsonify({
#         "text": text,
#         "sentiment": sentiment,
#         "polarity": round(compound, 3),
#         "confidence": f"{confidence}%",
#         "emoji": emoji_map[sentiment],
#         "positive_words": positive_words,
#         "negative_words": negative_words,
#         "tone": tone
#     })

# @app.route("/")
# def home():
#     return jsonify({"status": "Sentiment Analysis API running"}), 200

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


























from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# Emoji mapping for sentiment
emoji_map = {
    "Positive": "😊",
    "Negative": "😞",
    "Neutral": "😐"
}

def detect_tone(text):
    """Simple tone detection based on polarity."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return "Joy"
    elif polarity < -0.5:
        return "Anger"
    elif -0.5 <= polarity < 0:
        return "Sadness"
    else:
        return "Neutral"

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Determine sentiment
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Confidence score
    confidence = round(abs(polarity) * 100, 2)

    # Highlight words
    words = text.split()
    positive_words = [w for w in words if TextBlob(w).sentiment.polarity > 0]
    negative_words = [w for w in words if TextBlob(w).sentiment.polarity < 0]

    # Detect simple tone
    tone = detect_tone(text)

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "confidence": f"{confidence}%",
        "emoji": emoji_map[sentiment],
        "positive_words": positive_words,
        "negative_words": negative_words,
        "tone": tone
    })

@app.route("/")
def home():
    return jsonify({"status": "Sentiment Analysis API running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "polarity": round(polarity, 3)
    })

@app.route("/")
def home():
    return jsonify({"status": "Sentiment Analysis API running"}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)



# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- import CORS
from resume_utils import (
    load_file, split_into_sentences, build_embeddings,
    retrieve_best_sentence, run_genai_chain,
    route_question, get_top_k_context
)

app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes


@app.route("/query", methods=["POST"])
def query_resume():
    if 'file' not in request.files or 'question' not in request.form:
        return jsonify({"error": "File and question required"}), 400

    file = request.files['file']
    question = request.form['question']

    # ---------------- Load and preprocess resume ----------------
    text, error = load_file(file)
    if error:
        return jsonify({"error": error}), 400

    sentences = split_into_sentences(text)
    model, embeddings = build_embeddings(sentences)

    # ---------------- Decide mode ----------------
    mode = route_question(question)

    if mode == "semantic":
        # Semantic search using embeddings
        best_sentence, confidence = retrieve_best_sentence(question, model, embeddings, sentences)
        return jsonify({
            "mode": "semantic",
            "answer": best_sentence,
            "confidence": round(float(confidence), 2)
        })
    else:
        # GenAI: Use top-k context for better QA
        context = get_top_k_context(question, model, embeddings, sentences)
        genai_answer = run_genai_chain(sentences, question)
        return jsonify({
            "mode": "genai",
            "context_used": context,
            "answer": genai_answer
        })


@app.route("/")
def home():
    return jsonify({"status": "Resume Chatbot API running"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)







# # backend/app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # <-- import CORS
# from resume_utils import load_file, split_into_sentences, build_embeddings, retrieve_best_sentence, run_genai_chain

# app = Flask(__name__)
# CORS(app)  # <-- enable CORS for all routes


# @app.route("/query", methods=["POST"])
# def query_resume():
#     if 'file' not in request.files or 'question' not in request.form:
#         return jsonify({"error": "File and question required"}), 400

#     file = request.files['file']
#     question = request.form['question']

#     text, error = load_file(file)
#     if error:
#         return jsonify({"error": error}), 400

#     sentences = split_into_sentences(text)
#     model, embeddings = build_embeddings(sentences)

#     # Semantic search using embeddings
#     best_sentence, confidence = retrieve_best_sentence(question, model, embeddings, sentences)

#     # Optional: GenAI QA (LangChain)
#     genai_answer = run_genai_chain(sentences, question)

#     return jsonify({
#         "semantic_answer": best_sentence,
#         "semantic_confidence": round(float(confidence), 2),
#         "genai_answer": genai_answer
#     })

# @app.route("/")
# def home():
#     return jsonify({"status": "Resume Chatbot API running"}), 200

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=True)

