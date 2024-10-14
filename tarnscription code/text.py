import os
import speech_recognition as sr
from pydub import AudioSegment, effects
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to enhance audio quality
def preprocess_audio(chunk):
    # Normalize the audio and set a consistent sample rate and channels
    return effects.normalize(chunk.set_frame_rate(16000).set_channels(1))

# Function to transcribe a single audio chunk
def transcribe_chunk(chunk, index, file_path):
    temp_filename = f"temp_chunk_{index}.wav"
    transcription = ""
    try:
        chunk.export(temp_filename, format="wav")
        with sr.AudioFile(temp_filename) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio)
        logging.info(f"Successfully transcribed chunk {index} of {file_path}")
    except sr.UnknownValueError:
        logging.warning(f"Could not understand chunk {index} of {file_path}")
    except sr.RequestError as e:
        logging.error(f"Request error for chunk {index} of {file_path}; {e}")
    except Exception as e:
        logging.error(f"Error during transcription of chunk {index} of {file_path}: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    return transcription

# Function to split and transcribe an audio file
def transcribe_audio(file_path, chunk_duration_ms=60000):  # Chunk duration set to 60 seconds
    try:
        # Load the audio file
        sound = AudioSegment.from_file(file_path)
        # Split audio into chunks
        chunks = [sound[i:i + chunk_duration_ms] for i in range(0, len(sound), chunk_duration_ms)]
        transcriptions = []
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:  # Adjust max_workers based on chunk count
            futures = {executor.submit(transcribe_chunk, preprocess_audio(chunk), i, file_path): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    transcriptions.append(result if result else "")
                except Exception as e:
                    logging.error(f"Error with chunk {index} of {file_path}: {e}")
        return " ".join(transcriptions)
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return ""

# Function to transcribe all audio files in a directory
def transcribe_directory(directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    audio_formats = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff')
    for file in os.listdir(directory):
        if file.lower().endswith(audio_formats):
            file_path = os.path.join(directory, file)
            output_file = os.path.join(output_directory, f"{os.path.splitext(file)[0]}.txt")
            if os.path.exists(output_file):
                logging.info(f"Transcription for {file} already exists. Skipping...")
                continue
            logging.info(f"Transcribing {file_path}...")
            transcription = transcribe_audio(file_path)
            if transcription:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcription)
                logging.info(f"Saved transcription of {file} to {output_file}")

# Example usage
directory = r"F:\channel3 data\path\to\your\directory"
output_directory = r"videocontent\channel1"
transcribe_directory(directory, output_directory)
