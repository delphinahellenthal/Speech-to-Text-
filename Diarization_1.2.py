# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:55:31 2024

@author: rylie
"""

#https://huggingface.co/settings/tokens 

import os
from pyannote.audio import Pipeline
from huggingface_hub import hf_hub_download
from datetime import datetime

# Path to the folder containing WAV files
folder_path = '/Users/rylie/OneDrive/BAIS Capstone/Wav Files'
output_html_path = '/Users/rylie/OneDrive/BAIS Capstone/diarization_results.html'  # Specify a full file path

# Verify model access for 'pyannote/segmentation' before loading the pipeline
try:
    model_path = hf_hub_download(repo_id="pyannote/segmentation", filename="config.yaml", use_auth_token="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")
    print("Model access successful:", model_path)
except Exception as e:
    print("Model access error:", e)
    print("Could not load the PyAnnote model. Ensure your token is valid and you've accepted any model agreements.")
    exit()  # Stop execution if model access is not verified

# Load PyAnnote pipeline for speaker diarization
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    if pipeline is None:
        raise ValueError("Pipeline loading failed. Verify access and token permissions.")
except Exception as e:
    print("Could not load the PyAnnote pipeline. Ensure your token is valid and you've accepted any model agreements.")
    pipeline = None

# Proceed only if pipeline was loaded successfully
if pipeline:
    # Dictionary to store diarization results
    diarization_results = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            
            # Check if the file exists and is accessible
            if os.path.exists(file_path):
                try:
                    # Apply diarization pipeline to audio file
                    diarization = pipeline(file_path)
                    
                    # Capture speaker segments
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segment_info = {
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        }
                        segments.append(segment_info)
                    
                    diarization_results[filename] = segments

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"File not found: {file_path}")

    # Generate and save HTML content if there are results
    if diarization_results:
        html_content = "<html><body><h1>Speaker Diarization Results</h1>"
        for filename, segments in diarization_results.items():
            html_content += f"<h2>{filename}</h2><ul>"
            for segment in segments:
                start_time = datetime.utcfromtimestamp(segment["start"]).strftime('%H:%M:%S')
                end_time = datetime.utcfromtimestamp(segment["end"]).strftime('%H:%M:%S')
                html_content += f"<li>Speaker {segment['speaker']}: {start_time} - {end_time}</li>"
            html_content += "</ul>"
        html_content += "</body></html>"

        try:
            with open(output_html_path, "w") as file:
                file.write(html_content)
            print("Diarization results saved to HTML.")
        except PermissionError:
            print(f"Permission denied: Unable to write to '{output_html_path}'. Please check the file path.")
    else:
        print("No diarization results to save.")
else:
    print("Pipeline was not loaded. Please check your token and model access.")
