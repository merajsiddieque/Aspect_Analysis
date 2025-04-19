import os
import whisper
import ffmpeg

def convert_to_wav(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, ac=1, ar='16000').run(overwrite_output=True, quiet=True)
        print(f"Converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        print("Error during audio conversion:", e)
        return False
    return True

def transcribe_audio(audio_path, output_txt_path):
    # Load the Whisper large model for better accuracy
    model = whisper.load_model("base")
    print("Large model loaded.")

    # Transcribe the audio with Hindi language specified
    result = model.transcribe(audio_path, language="hi")
    print("Transcription complete.")

    # Save the result to a text file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcript saved to {output_txt_path}")

def main():
    input_path = r"C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\web\input_audio.webm"
    wav_path = r"C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\web\input_audio_converted.wav"
    output_txt_path = r"C:\Users\meraj\OneDrive\Desktop\Sentiment_Analysis\web\asr-text.txt"

    if convert_to_wav(input_path, wav_path):
        transcribe_audio(wav_path, output_txt_path)

if __name__ == "__main__":
    main()
