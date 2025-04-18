from flask import Flask, request, send_file, send_from_directory, render_template
from flask_cors import CORS
import os
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load
import fasttext

# Set up Flask app
app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = BASE_DIR
PREDICTION_FILE = os.path.join(BASE_DIR, 'test-predictions.txt')
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Model2'))
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'fasttext', 'lid.176.ftz')

# Load FastText language detection model once
lang_detector = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Helper: Read file lines
def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]

# Helper: Save list to file
def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))

# Helper: Load pickled model/vectorizer 
def loadObjectFromFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)

# Detect dominant language using FastText
def detect_language(text_lines):
    label_counts = {}
    for line in text_lines:
        if not line.strip():
            continue
        prediction = lang_detector.predict(line.strip())[0][0]  # e.g. '__label__hi'
        lang = prediction.replace('__label__', '')
        label_counts[lang] = label_counts.get(lang, 0) + 1
    return max(label_counts, key=label_counts.get)

# Helper function to load models for each language
def load_language_models(language):
    """
    Given a language code (e.g., 'hi', 'mr', 'te'), load the corresponding models.
    """
    lang_map = {
        'hi': 'hindi',
        'mr': 'marathi',
        'te': 'telugu'
    }
    
    if language not in lang_map:
        raise Exception(f"Unsupported or untrained language: {language}")
    
    prefix = lang_map[language]

    # Load the word vectorizer, char vectorizer, and classifier for the specified language
    wordVectPath = os.path.join(MODEL_DIR, f"{prefix}-train-vect-word-svm.pkl")
    charVectPath = os.path.join(MODEL_DIR, f"{prefix}-train-vect-char-svm.pkl")
    classifierPath = os.path.join(MODEL_DIR, f"{prefix}-classifier-svm.pkl")

    wordVect = loadObjectFromFile(wordVectPath)
    charVect = loadObjectFromFile(charVectPath)
    clf = loadObjectFromFile(classifierPath)

    return wordVect, charVect, clf

# Predict based on detected language
def predict(testFilePath, outputPredPath):
    # Read uploaded test data
    testData = readLinesFromFile(testFilePath)

    # Step 1: Detect dominant language
    dominant_lang = detect_language(testData)
    print(f"[DEBUG] Detected language: {dominant_lang}")

    # Step 2: Load appropriate models based on detected language
    wordVect, charVect, clf = load_language_models(dominant_lang)

    # Step 3: Transform and predict
    wordTestTfIdf = wordVect.transform(testData)
    charTestTfIdf = charVect.transform(testData)
    combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf])

    predictions = [str(label) for label in clf.predict(combinedTestTfIdf)]
    writeListToFile(outputPredPath, predictions)
    print(f"[DEBUG] Predictions saved to {outputPredPath}")

# Serve the main HTML page
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    test_file_path = os.path.join(UPLOAD_FOLDER, 'unlabeled_reviews.txt')
    file.save(test_file_path)

    print(f"[DEBUG] File saved at: {test_file_path}")
    print(f"[DEBUG] Running prediction...")

    try:
        predict(test_file_path, PREDICTION_FILE)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return 'Prediction failed', 500

    if not os.path.exists(PREDICTION_FILE):
        print(f"[ERROR] File not found: {PREDICTION_FILE}")
        return 'Prediction file not found', 500

    print(f"[DEBUG] Prediction done. Sending file back.")
    return send_file(PREDICTION_FILE, as_attachment=True, download_name='predictions.txt')

# Serve static files like CSS and JS
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
