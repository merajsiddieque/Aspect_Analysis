import os
import requests
import zipfile

def download_hindi_model(model_url, download_path):
    # Send a GET request to download the model
    print(f"Downloading Hindi model from {model_url}...")
    response = requests.get(model_url, stream=True)

    if response.status_code == 200:
        # Save the model zip file to the specified path
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Model downloaded successfully to {download_path}")

        # Extract the zip file
        print("Extracting model...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(download_path))
        print(f"Model extracted to {os.path.dirname(download_path)}")
    else:
        print("Failed to download the model. Please check the URL or your internet connection.")

def main():
    # URL for the Vosk Hindi model
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.15.zip"
    
    # Path to save the downloaded zip file
    download_path = r"C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\vosk-model-small-hi-0.15.zip"
    
    # Download and extract the Hindi model
    download_hindi_model(model_url, download_path)

if __name__ == "__main__":
    main()
