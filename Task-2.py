import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

def recognize_speech_sr(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Could not request results, please check your internet connection."

def recognize_speech_wav2vec(audio_file):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    audio, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

if __name__ == "__main__":
    audio_path = "ff.wav" 
    print("SpeechRecognition Output:", recognize_speech_sr(audio_path))
    print("Wav2Vec2 Output:", recognize_speech_wav2vec(audio_path))
