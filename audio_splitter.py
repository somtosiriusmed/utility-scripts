#!/usr/bin/env python3
"""
Audio Splitter Script - Python 3.13 Compatible
Splits MP3 files into smaller WAV segments of 5-8 seconds each.
Uses librosa and soundfile instead of pydub for Python 3.13 compatibility.
"""

import os
import glob
import random
import librosa
import soundfile as sf
import numpy as np

def split_audio_files(input_folder, output_folder=None, min_duration=5, max_duration=8):
    """
    Split MP3 files into smaller WAV segments.
    
    Args:
        input_folder (str): Path to folder containing MP3 files
        output_folder (str): Path to output folder (optional, defaults to input_folder/split_audio)
        min_duration (int): Minimum segment duration in seconds
        max_duration (int): Maximum segment duration in seconds
    """
    
    # Set default output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "split_audio")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all MP3 files in the input folder
    mp3_files = glob.glob(os.path.join(input_folder, "*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {input_folder}")
        return
    
    print(f"Found {len(mp3_files)} MP3 files to process...")
    
    total_segments = 0
    
    for mp3_file in mp3_files:
        print(f"\nProcessing: {os.path.basename(mp3_file)}")
        
        try:
            # Load the audio file using librosa
            audio_data, sample_rate = librosa.load(mp3_file, sr=None)
            
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(mp3_file))[0]
            
            # Calculate total duration in seconds
            total_duration = len(audio_data) / sample_rate
            print(f"  Total duration: {total_duration:.1f} seconds")
            
            segment_count = 0
            start_sample = 0
            
            while start_sample < len(audio_data):
                # Random duration between min and max seconds
                segment_duration = random.uniform(min_duration, max_duration)
                
                # Convert duration to samples
                segment_samples = int(segment_duration * sample_rate)
                
                # Ensure we don't go beyond the audio length
                end_sample = min(start_sample + segment_samples, len(audio_data))
                
                # Skip very short segments at the end (less than 2 seconds)
                actual_duration = (end_sample - start_sample) / sample_rate
                if actual_duration < 2.0:
                    break
                
                # Extract the segment
                segment = audio_data[start_sample:end_sample]
                
                # Create output filename
                output_filename = f"{base_name}_segment_{segment_count:03d}.wav"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save as WAV using soundfile
                sf.write(output_path, segment, sample_rate)
                
                print(f"  Created: {output_filename} ({actual_duration:.1f}s)")
                
                segment_count += 1
                total_segments += 1
                start_sample = end_sample
            
            print(f"  Generated {segment_count} segments from {os.path.basename(mp3_file)}")
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(mp3_file)}: {str(e)}")
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"Total segments created: {total_segments}")
    print(f"Output folder: {output_folder}")

def main():
    """Main function to run the audio splitter."""
    
    # Configuration - modify these as needed
    INPUT_FOLDER = "downloads"
    
    # Remove quotes if user included them
    INPUT_FOLDER = INPUT_FOLDER.strip('"\'')
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' does not exist!")
        return
    
    # Optional: Ask for custom output folder
    custom_output = "output_audio"
    output_folder = custom_output if custom_output else None
    
    # Optional: Ask for custom duration range
    try:
        min_dur = 5
        min_duration = float(min_dur) if min_dur else 5
        
        max_dur = 8
        max_duration = float(max_dur) if max_dur else 8
        
        if min_duration >= max_duration:
            print("❌ Error: Minimum duration must be less than maximum duration!")
            return
            
    except ValueError:
        print("❌ Error: Please enter valid numbers for duration!")
        return
    
    print(f"\n🎵 Starting audio splitting...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Segment duration: {min_duration}-{max_duration} seconds")
    
    # Run the splitter
    split_audio_files(INPUT_FOLDER, output_folder, min_duration, max_duration)

if __name__ == "__main__":
    main()
