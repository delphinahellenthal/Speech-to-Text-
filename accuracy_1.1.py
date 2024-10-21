#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:25:48 2024

@author: chloekalb
"""
import speech_recognition as sr
import docx
import re
from pydub import AudioSegment
from textblob import TextBlob
import os

# Function to read the actual key from a DOCX file
def read_docx(file_path):
    """
    Reads text from a DOCX file.
    :param file_path: Path to the DOCX file.
    :return: Text content of the DOCX file as a string.
    """
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
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Function to calculate accuracy
def calculate_accuracy(transcribed_words, reference_words):
    matches = sum(1 for transcribed_word, reference_word in zip(transcribed_words, reference_words) if transcribed_word == reference_word)
    total_words = len(reference_words)
    return (matches / total_words * 100) if total_words > 0 else 0

# Function to preprocess and downsample the audio file
def preprocess_audio(file_path):
    try:
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(16000)
        normalized_audio = audio.normalize()
        processed_file_path = os.path.join(os.path.dirname(file_path), 'processed_audio.wav')
        normalized_audio.export(processed_file_path, format="wav")
        return processed_file_path
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return ""

# Function to transcribe the audio using Google Web Speech API
def transcribe_audio_google(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Google error: {e}")
        return ""

# Function to post-process transcription using TextBlob for basic spell correction
def correct_transcription(text):
    return str(TextBlob(text).correct())

# Main function to run the comparison and calculate the accuracy
def main():
    wav_file = '/Users/chloekalb/Downloads/Aflac Data/WAV/Call-2_converted.wav'  # Update to your audio file path
    docx_file = '/Users/chloekalb/Downloads/Aflac Data/DOCX/Call 2 Key.docx'  # Update to your DOCX file path

    reference_text = read_docx(docx_file)
    reference_words = preprocess_text(reference_text)
    processed_wav_file = preprocess_audio(wav_file)
    transcribed_text = transcribe_audio_google(processed_wav_file)

    corrected_transcription = correct_transcription(transcribed_text)
    transcribed_words = preprocess_text(corrected_transcription)
    accuracy = calculate_accuracy(transcribed_words, reference_words)

    print(f"Transcribed Text: {corrected_transcription}")
    print(f"Reference Text: {reference_text}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
