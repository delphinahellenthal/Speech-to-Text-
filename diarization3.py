#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:19:00 2024

@author: delphinahellenthal
"""

#AutoTokenizer

#import torch
#import torchaudio
#import torchvision
#import transformers
#import tokenizers
#from datetime import datetime


from huggingface_hub import HfApi
from pyannote.audio import Pipeline
import os


# List available pyannote pipelines
available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]
print(list(filter(lambda p: p.startswith("pyannote/"), available_pipelines)))

# Set the token for HuggingFace
token = "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"

# Load the pre-trained pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)

# Directory where your .wav files are stored
input_dir = "/Users/delphinahellenthal/Desktop/BAISCapstone/bais_env/lib/python3.13/"
output_dir = "/Users/delphinahellenthal/Desktop/BAISCapstone/bais_env/lib/python3.13/diarization_output/"
os.makedirs(output_dir, exist_ok=True)

# Function to generate HTML for diarization result
def generate_html(diarization, filename):
    html_content = """
    <html>
    <head>
        <title>Diarization Results</title>
        <style>
            body {font-family: Arial, sans-serif;}
            table {width: 100%; border-collapse: collapse; margin-top: 20px;}
            th, td {padding: 8px; text-align: left; border: 1px solid #ddd;}
            th {background-color: #f2f2f2;}
            h1 {color: #4CAF50;}
        </style>
    </head>
    <body>
        <h1>Diarization Results for {filename}</h1>
        <table>
            <tr>
                <th>Speaker</th>
                <th>Start Time (s)</th>
                <th>End Time (s)</th>
            </tr>
    """.format(filename=filename)
    
    # Iterate through the diarization and add to HTML
    for speech_segment in diarization.itersegments():
        speaker = diarization[speech_segment]
        start_time = round(speech_segment.start, 2)
        end_time = round(speech_segment.end, 2)
        
        html_content += """
            <tr>
                <td>{speaker}</td>
                <td>{start_time}</td>
                <td>{end_time}</td>
            </tr>
        """.format(speaker=speaker, start_time=start_time, end_time=end_time)

    # Closing tags
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Return HTML content
    return html_content

# Iterate through all files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        try:
            # Full path to the audio file
            audio_file = os.path.join(input_dir, filename)

            # Apply the pipeline to the audio file
            diarization = pipeline(audio_file)

            # Generate the HTML file name (same as audio file but with .html extension)
            html_file = os.path.join(output_dir, filename.replace(".wav", ".html"))

            # Generate HTML content from diarization
            html_content = generate_html(diarization, filename)

            # Save the HTML content to a file
            with open(html_file, "w") as html:
                html.write(html_content)

            print(f"Diarization for {filename} completed and saved as {html_file}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
