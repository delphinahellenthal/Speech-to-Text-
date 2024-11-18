#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:27:10 2024

@author: chloekalb
version 1.3
"""
import speech_recognition as sr
import docx
import re
from pydub import AudioSegment
from textblob import TextBlob
import os
import jiwer  # For calculating WER
from fuzzywuzzy import fuzz
from vosk import Model, KaldiRecognizer
import wave
import json
from pydub import AudioSegment, effects
import numpy as np
import pydub

# Function to read the actual key from a DOCX file
def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""

# Function to preprocess text: lowercase, remove punctuation, etc.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

def equalize_audio(audio):
    # Step 1: Apply high-pass filter to remove frequencies below `low_cutoff`
    low_band = audio.low_pass_filter(250).apply_gain(-10)
    
    # Step 2: Boost mid-range frequencies (around `mid_boost` Hz)
    low_mid_band = audio.high_pass_filter(250).low_pass_filter(1000).apply_gain(3)
    
    # Combine original audio with boosted mid-range frequencies
    mid_band = audio.high_pass_filter(1000).low_pass_filter(4000).apply_gain(5)
    
    # Step 3: Apply low-pass filter to remove frequencies above `high_cutoff`
    high_band = audio.high_pass_filter(4000).low_pass_filter(8000).apply_gain(5)
    
    very_high_band = audio.high_pass_filter(8000).apply_gain(-5)

    # Overlay the bands for the final equalized audio
    equalized_audio = low_band.overlay(low_mid_band).overlay(mid_band).overlay(high_band).overlay(very_high_band)
    
    return equalized_audio



# Updated preprocess_audio function to include equalization
def preprocess_audio(file_path):
    # Load the audio file
  
    try: 
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(16000)
        normalized_audio = audio.normalize()
        equalized_audio = equalize_audio(normalized_audio)
        processed_file_path = os.path.join(os.path.dirname(file_path),'processed_' + os.path.basename(file_path))
        equalized_audio.export(processed_file_path, format = "wav")
        return processed_file_path
    except Exception as e: 
        print(f"Error processing audio file: {e}")
    return ""

# Function to transcribe audio using Vosk (using the processed audio)
import io

def transcribe_audio_vosk(audio, model):
    try:
        # Convert the audio to raw data for Vosk to process
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        with wave.open(buffer, "rb") as wf:
            recognizer = KaldiRecognizer(model, wf.getframerate())
            transcription = ""

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcription += result.get("text", "") + " "

            final_result = json.loads(recognizer.FinalResult())
            transcription += final_result.get("text", "")
            return transcription.strip()
    except Exception as e:
        print(f"Error transcribing audio with Vosk: {e}")
        return ""



# Function to post-process transcription using TextBlob for basic spell correction
def correct_transcription(text):
    return str(TextBlob(text).correct())

# Function to calculate Word Error Rate (WER)
def calculate_wer(transcribed_text, reference_text):
    wer_score = jiwer.wer(reference_text, transcribed_text)
    return wer_score * 100  # Convert to percentage

# Function to calculate fuzzy similarity score
def calculate_fuzzy_score(transcribed_text, reference_text):
    return fuzz.ratio(transcribed_text, reference_text)

def main():
    wav_folder = '/Users/chloekalb/Downloads/Aflac Data/WAV'
    docx_folder = '/Users/chloekalb/Downloads/Aflac Data/DOCX'
    
    vosk_model_path = "/Users/chloekalb/Downloads/vosk-model-en-us-0.22"  # Update this to your model path
    model = Model(vosk_model_path)

    wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
    docx_files = [f for f in os.listdir(docx_folder) if f.endswith('.docx')]

    print("Available DOCX files:")
    for docx_file in docx_files:
        print(docx_file)
        
    print("Available WAV files:")
    for wav_file in wav_files:
        print(wav_file)
            

    # Initialize variables to track total WER and fuzzy score
    total_wer = 0
    total_fuzzy = 0
    total_files_processed = 0

    for wav_filename in wav_files:
        wav_file_path = os.path.join(wav_folder, wav_filename)
        docx_filename = wav_filename.replace('-', ' ').replace('_converted.wav', ' Key.docx')
        docx_file_path = os.path.join(docx_folder, docx_filename)

        if os.path.exists(docx_file_path):
            print(f"\nProcessing {wav_filename} with {docx_filename}")

            # Read the reference text and preprocess it
            reference_text = read_docx(docx_file_path)
            reference_text_processed = ' '.join(preprocess_text(reference_text))

            # Preprocess, transcribe, and correct transcription
            processed_wav_file = preprocess_audio(wav_file)
            transcribed_text = transcribe_audio_vosk(processed_wav_file, model)
            corrected_transcription = correct_transcription(transcribed_text) if transcribed_text else ""

            # Print the reference text and transcription for verification
            print("\n--- Reference Text ---")
            print(reference_text)
            print("\n--- Transcription ---")
            print(corrected_transcription)

            if corrected_transcription:
                # Calculate WER and Fuzzy Score
                wer_score = calculate_wer(corrected_transcription, reference_text_processed)
                fuzzy_score = calculate_fuzzy_score(corrected_transcription, reference_text_processed)
                
                total_wer += wer_score
                total_fuzzy += fuzzy_score
                total_files_processed += 1

                print(f"\n--- Results for {wav_filename} ---")
                print(f"WER Accuracy: {100 - wer_score:.2f}%")
                print(f"Fuzzy Similarity Score: {fuzzy_score}%")
            else:
                print(f"Error: No transcription available for {wav_filename}")
        else:
            print(f"Matching DOCX file not found for {wav_filename}")
            
    if total_files_processed > 0:
         average_wer = total_wer / total_files_processed
         average_fuzzy = total_fuzzy / total_files_processed
         print(f"\n--- Summary ---")
         print(f"Processed {total_files_processed} files.")
         print(f"Average WER Accuracy: {100 - average_wer:.2f}%")
         print(f"Average Fuzzy Similarity Score: {average_fuzzy:.2f}%")
    else:
         print("No files were processed.")

if __name__ == "__main__":
    main()
    
