from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from pickle import load
import os

# Helper: Read lines from file and return both text and labels
def read_lines_with_labels(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # Split the line by spaces and assume the last word is the label
                words = line.split(' ')
                label = words[-1]  # Last word is the label
                text = ' '.join(words[:-1])  # All other words are the text
                texts.append(text)
                labels.append(label)
    print(f"[DEBUG] Found {len(texts)} texts and {len(labels)} labels.")
    return texts, labels


# Helper: Load pickled model/vectorizer 
def load_object_from_file(file_path):
    with open(file_path, 'rb') as file:
        return load(file)

# Helper: Transform test data with TF-IDF
def create_test_tfidf(test_data, tfidf_vect):
    return tfidf_vect.transform(test_data)

# Core logic: Predict labels and write results to file
def predict(test_file_path, output_pred_path):
    # Update paths to the models
    word_vect_path = r'C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\Model2\train-vect-word-svm.pkl'
    char_vect_path = r'C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\Model2\train-vect-char-svm.pkl'
    classifier_path = r'C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\Model2\classifier-svm.pkl'

    # Load models
    word_vect = load_object_from_file(word_vect_path)
    char_vect = load_object_from_file(char_vect_path)
    clf = load_object_from_file(classifier_path)

    # Read test data and labels
    test_data, actual_labels = read_lines_with_labels(test_file_path)

    # Vectorize the test data
    word_test_tfidf = create_test_tfidf(test_data, word_vect)
    char_test_tfidf = create_test_tfidf(test_data, char_vect)

    # Combine word and char features
    combined_test_tfidf = hstack([word_test_tfidf, char_test_tfidf])

    # Predict the labels for the test data
    predictions = clf.predict(combined_test_tfidf)

    # Print actual labels and predictions for debugging
    print(f"Actual Labels: {actual_labels}")
    print(f"Predictions: {predictions}")

    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Write predictions to file
    with open(output_pred_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(predictions))

# Example usage
test_file_path = r'C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\datasets\test-data.txt'
output_pred_path = r'C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\Score\test_predictions.txt'
predict(test_file_path, output_pred_path)
