import os
import whisper
import ffmpeg
from flask import Flask, request, send_file, send_from_directory, jsonify
from flask_cors import CORS
from scipy.sparse import hstack
from pickle import load
import fasttext

# Initialize Flask app
app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = BASE_DIR  # Points to the 'web' directory
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Model2'))

# File paths
PREDICTION_FILE = os.path.join(UPLOAD_FOLDER, 'test-predictions.txt')
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'fasttext', 'lid.176.ftz')

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
lang_detector = fasttext.load_model(FASTTEXT_MODEL_PATH)
whisper_model = whisper.load_model("base")

# Helper functions
def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def write_lines(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return load(f)

def detect_language(lines):
    counts = {}
    for line in lines:
        pred = lang_detector.predict(line)[0][0].replace('__label__', '')
        if pred in ['hi', 'mr', 'te']:
            counts[pred] = counts.get(pred, 0) + 1
    if not counts:
        raise Exception("No supported language detected in the input.")
    return max(counts, key=counts.get)

def load_hindi_models():
    """Load Hindi models specifically for audio processing"""
    word_vect = load_pickle(os.path.join(MODEL_DIR, "train-vect-word-svm.pkl"))
    char_vect = load_pickle(os.path.join(MODEL_DIR, "train-vect-char-svm.pkl"))
    clf = load_pickle(os.path.join(MODEL_DIR, "classifier-svm.pkl"))
    return word_vect, char_vect, clf

def load_models(language):
    """Load language-specific models for text processing"""
    lang_map = {'hi': 'hindi', 'mr': 'marathi', 'te': 'telugu'}
    prefix = lang_map[language]
    word_vect = load_pickle(os.path.join(MODEL_DIR, f"{prefix}-train-vect-word-svm.pkl"))
    char_vect = load_pickle(os.path.join(MODEL_DIR, f"{prefix}-train-vect-char-svm.pkl"))
    clf = load_pickle(os.path.join(MODEL_DIR, f"{prefix}-classifier-svm.pkl"))
    return word_vect, char_vect, clf

def predict_sentiment(lines, word_vect, char_vect, clf):
    """Core prediction function used by both text and audio paths"""
    word_tf = word_vect.transform(lines)
    char_tf = char_vect.transform(lines)
    combined_tf = hstack([word_tf, char_tf])
    return clf.predict(combined_tf)

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format"""
    try:
        ffmpeg.input(input_path).output(output_path, ac=1, ar='16000').run(
            overwrite_output=True, quiet=True)
        return True
    except ffmpeg.Error as e:
        print("Audio conversion error:", e)
        return False

def transcribe_audio(audio_path, output_txt_path=None):
    """Transcribe audio using Whisper and write each word on a new line"""
    result = whisper_model.transcribe(audio_path, language="hi")
    transcription = result["text"]
    if output_txt_path:
        # Split transcription into words and write each on a new line
        words = transcription.split()
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(words))
    return transcription

# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/upload-asr', methods=['POST'])
def process_asr_text():
    """Process transcribed text from audio (Hindi only)"""
    try:
        input_path = os.path.join(UPLOAD_FOLDER, 'asr-text.txt')
        output_path = os.path.join(UPLOAD_FOLDER, 'test-predictions.txt')
        
        # Load Hindi models directly (no language detection needed)
        word_vect, char_vect, clf = load_hindi_models()
        
        lines = read_lines(input_path)
        preds = predict_sentiment(lines, word_vect, char_vect, clf)
        write_lines(output_path, [str(p) for p in preds])
        
        return send_file(output_path, as_attachment=True, download_name='asr-predictions.txt')
    except Exception as e:
        print(f"ASR processing error: {e}")
        return "Processing failed", 500

@app.route('/upload', methods=['POST'])
def handle_text_upload():
    """Process text file upload with language detection"""
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    text_path = os.path.join(UPLOAD_FOLDER, 'unlabeled_reviews.txt')
    file.save(text_path)

    try:
        # Detect language first using FastText
        lines = read_lines(text_path)
        language = detect_language(lines)
        
        # Load appropriate language models
        word_vect, char_vect, clf = load_models(language)
        
        # Predict sentiment
        preds = predict_sentiment(lines, word_vect, char_vect, clf)
        write_lines(PREDICTION_FILE, [str(p) for p in preds])
        
        return send_file(PREDICTION_FILE, as_attachment=True, download_name='predictions.txt')
    except Exception as e:
        print(f"Text processing error: {e}")
        return 'Processing failed', 500

@app.route('/upload-audio', methods=['POST'])
def handle_audio_upload():
    """Handle audio file upload and processing"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save uploaded audio
        webm_path = os.path.join(UPLOAD_FOLDER, 'input_audio.webm')
        audio_file.save(webm_path)
        
        # Convert to WAV
        wav_path = os.path.join(UPLOAD_FOLDER, 'input_audio_converted.wav')
        if not convert_to_wav(webm_path, wav_path):
            return jsonify({'error': 'Audio conversion failed'}), 500
        
        # Transcribe audio
        txt_path = os.path.join(UPLOAD_FOLDER, 'asr-text.txt')
        transcription = transcribe_audio(wav_path, txt_path)
        
        return jsonify({
            'message': 'Audio processed successfully',
            'transcription': transcription
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)