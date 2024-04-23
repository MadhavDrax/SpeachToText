import gradio as gr 
import numpy as np
import whisper

def transcribe(audio):
    
    # First- we select the model for whisper AI
    # Here we are using base model because it is fast but it dose not have good accuracy in non famous languages 
    model=whisper.load_model("base")
    
    #Second- we load audio in our database and pad/trim audio to fit in 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Third- make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Fourth- we find the probability and find out the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Fifth- now we decode the audio and convert it into text
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

# Now we build a interface by the help of Gradio web UI 
gr.Interface(
    title = 'Computer Vision Project By Using Whisper AI With The Help Of Gradio Web UI', 
    fn=transcribe, 
    inputs=[
        gr.Audio(sources="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch(share=True)

#The End