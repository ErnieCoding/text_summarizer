import whisper, torch

def transcribe_meeting():
   filepath = "audio_recordings/new_meeting.mp3"

   print("STARTING TRANSCRIPTION")

   print(f"\nPyTorch version: {torch.__version__}\n")
   print(f"\nCUDA version: {torch.version.cuda}\n")
   print(f"\nCUDA available: {torch.cuda.is_available()}\n")
   print(f"\ncuDNN enabled: {torch.backends.cudnn.enabled}\n")

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = whisper.load_model("large").to(device)

   result = model.transcribe(filepath, fp16 = (device == "cuda"))

   with open("transcripts/new_meeting.txt", "w+", encoding="utf-8") as file:
      file.write(result["text"])

if __name__ == "__main__":
   transcribe_meeting()