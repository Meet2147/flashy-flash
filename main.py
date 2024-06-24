import openai
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Audio recording parameters
AUDIO_FILE_PATH = 'recorded_audio.wav'
SAMPLE_RATE = 16000

class RecordRequest(BaseModel):
    duration: int

class FlashcardsResponse(BaseModel):
    question: str
    answer: str

# Function to record audio
def record_audio(duration, sample_rate, filename):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write(filename, sample_rate, audio_data)  # Save as WAV file
    print("Recording finished")

# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",  # Use the appropriate model name
            file=audio_file
        )
    transcribed_text = response['text']
    return transcribed_text

# Function to extract key concepts and generate Q&A pairs using OpenAI GPT-3.5 Turbo API
def generate_flashcards(transcribed_text):
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that helps create educational content."
        },
        {
            "role": "user",
            "content": f"Extract key concepts and definitions from the following text and format them as Q&A pairs for flashcards:\n\n{transcribed_text}"
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500  # Increase max tokens for better results
    )
    qa_pairs = response['choices'][0]['message']['content'].strip()
    return qa_pairs

# Function to format Q&A pairs into flashcards and save as JSON
def format_flashcards(qa_pairs):
    flashcards = []
    # Use regular expression to find Q&A pairs
    pairs = re.findall(r'(.+?\?)\s*(.+)', qa_pairs)
    
    for question, answer in pairs:
        flashcards.append({
            'question': question.strip(),
            'answer': answer.strip()
        })
    
    return flashcards

@app.post("/generate_flashcards", response_model=list[FlashcardsResponse])
async def generate_flashcards_endpoint(record_request: RecordRequest):
    duration = record_request.duration
    try:
        # Step 1: Record the audio
        record_audio(duration, SAMPLE_RATE, AUDIO_FILE_PATH)
        
        # Step 2: Transcribe the audio using Whisper API
        transcribed_text = transcribe_audio(AUDIO_FILE_PATH)
        
        # Step 3: Extract key concepts and generate Q&A pairs
        qa_pairs = generate_flashcards(transcribed_text)
        
        # Step 4: Format Q&A pairs into flashcards
        flashcards = format_flashcards(qa_pairs)
        
        return flashcards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
