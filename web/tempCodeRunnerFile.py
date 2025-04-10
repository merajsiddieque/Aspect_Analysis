from flask import Flask, request, send_file, send_from_directory, render_template
from flask_cors import CORS
import os
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load

# Set up Flask app
app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Use absolute path to avoid "file not found" issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = BASE_DIR
PREDICTION_FILE = os.path.join(BASE_DIR, 'test-predictions.txt')
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Model2'))


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

# Helper: Transform test data
def createTestTfIdf(testData, tfIdfVect):
    return tfIdfVect.transform(testData)

# Core logic: Predict labels and write to file
def predict(testFilePath, outputPredPath):
    wordVectPath = os.path.join(MODEL_DIR, "train-vect-word-svm.pkl")
    charVectPath = os.path.join(MODEL_DIR, "train-vect-char-svm.pkl")
    classifierPath = os.path.join(MODEL_DIR, "classifier-svm.pkl")

    # Load models
    wordVect = loadObjectFromFile(wordVectPath)
    charVect = loadObjectFromFile(charVectPath)
    clf = loadObjectFromFile(classifierPath)

    # Read uploaded test data
    testData = readLinesFromFile(testFilePath)

    # Vectorize
    wordTestTfIdf = createTestTfIdf(testData, wordVect)
    charTestTfIdf = createTestTfIdf(testData, charVect)

    # Combine word and char features
    combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf])

    # Predict and save results
    predictions = [str(label) for label in clf.predict(combinedTestTfIdf)]
    writeListToFile(outputPredPath, predictions)

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

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
