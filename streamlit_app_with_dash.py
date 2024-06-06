
from datetime import datetime
import streamlit as st
#---------------------------
import cv2
import cvzone
from PIL import Image
from ultralytics import YOLO
import torch
from transformers import pipeline
#---------------------------
import librosa
from transformers import Wav2Vec2FeatureExtractor
import pyaudio
import wave
import numpy as np
#---------------------------
#---------------------------
import numpy as np
import pandas as pd
import sounddevice as sd
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
# Initialize a Counter to store emotion counts
from collections import Counter
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#------------------------------
from google.cloud import speech
import streamlit as st
import numpy as np
import queue
import io
import os
import pyaudio
import re
import sys


# Set the page configuration at the beginning of the script
st.set_page_config(page_title="Emotion Analysis App", layout="wide")


page = st.sidebar.selectbox("Choose a page", ["Emotion Detection", "Dashboard"])

# Initialize models and processors
emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-latest")
text_emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-latest")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# emotion_model = torch.load('ser_modelv2_oversampled.pth', map_location=device)
# emotion_model.to(device)
text_emotion_model.to(device)



# emotion_model.eval()
# transcription_model.eval()
text_emotion_model.eval()




# Verify model files
model_files = ["ser_modelv2_oversampled.pth", "yolov8l-face.pt"]
for file in model_files:
    if not os.path.exists(file):
        st.error(f"Model file {file} does not exist")
    else:
        st.success(f"Model file {file} found")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the emotion model
try:
    emotion_model = torch.load('ser_modelv2_oversampled.pth', map_location=device)
    emotion_model.to(device)
    emotion_model.eval()
except Exception as e:
    st.error(f"Error loading emotion model: {e}")

# Load the face model
try:
    face_model = YOLO('yolov8l-face.pt')
except Exception as e:
    st.error(f"Error loading face model: {e}")

# Verify the model directory
model_path = 'facial_emotions_image_detection/checkpoint-3136'
if not os.path.exists(model_path):
    st.error(f"Model directory {model_path} does not exist")
else:
    st.success(f"Model directory {model_path} found")

# Load the image classification pipeline
try:
    pipe = pipeline('image-classification', model=model_path, device=-1)
except Exception as e:
    st.error(f"Error loading image classification pipeline: {e}")




if 'emotions_data' not in st.session_state:
    # st.session_state.emotions_data = pd.DataFrame(columns=['Face Emotions', 'Voice Emotions', 'Text Emotions'])
    # st.session_state.emotions_data = pd.DataFrame(columns=['Face Emotions'])
    st.session_state.emotions_data = pd.DataFrame()




def analyze_facial_emotion(frame, pipe, face_model):
    # Define the scale factor to reduce frame size by 50%
    scale_factor = 0.5
    face_result = face_model.predict(frame, conf=0.40)  # YOLOv8 face detection
    all_face_emotions = []

    for info in face_result:
        # Assuming `info` has the attribute `boxes` with face coordinates
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
            # Crop face from the frame
            face_img = frame[y1:y2, x1:x2]
            face_img_pil = Image.fromarray(face_img)  # Convert to PIL image for `pipe`
                
            # Predict emotion on the cropped face
            emotion_result = pipe(face_img_pil)
            emotion_text = emotion_result[0]['label']  # Assuming the highest confidence result
                
            all_face_emotions.append(emotion_text)

            # Draw rectangle and emotion text
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=9, rt=0)
            cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return all_face_emotions
#-----------------------------------------------------------------------------------

text_emotion_remap = {
    'anger': 'angry',
    'joy': 'happy',
    'sadness': 'sad',
    # Add other mappings if necessary
}


def is_silent(audio_chunk, threshold=0.01):
    """
    Determine if the audio chunk is silent based on a threshold.
    The threshold value can be adjusted based on testing and the sensitivity required.
    """
    return np.mean(np.abs(audio_chunk)) < threshold

def capture_audio_chunk(duration, sample_rate=16000):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return np.squeeze(recording)  # Convert from 2d to 1d array


    # Setup the request for Google 
# Assuming Google credentials are set in the environment or you can set them here
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pennine-transcription-536142f6b874.json"


def generate_unique_filename(base_name="recorded_audio", extension="wav"):
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.{extension}"



class MicrophoneStream:
    def __init__(self, rate, chunk, output_filename=None):
        self.rate = rate
        self.chunk = chunk
        self.output_filename = output_filename or generate_unique_filename()
        self.buff = queue.Queue()
        self.closed = True
        self.stop_flag = False

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.wave_file = wave.open(self.output_filename, 'wb')
        self.wave_file.setnchannels(1)
        self.wave_file.setsampwidth(self.audio_interface.get_sample_size(pyaudio.paInt16))
        self.wave_file.setframerate(self.rate)
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.fill_buffer
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.wave_file.close()
        self.audio_interface.terminate()
        self.closed = True
        self.buff.put(None)

    def fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self.buff.put(in_data)
        self.wave_file.writeframes(in_data)
        return None, pyaudio.paContinue

    def stop(self):
        self.stop_flag = True  # Set the stop flag to True to stop the generator

    def generator(self):
        while not self.closed and not self.stop_flag:
            chunk = self.buff.get()
            if chunk is None:
                break
            yield chunk




def listen_print_loop(responses, stream):
    num_chars_printed = 0
    complete_transcription = ""
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            # Clear the current line and print the interim result
            # st.write(transcript + overwrite_chars + '\r', end='', flush=True)
            num_chars_printed = len(transcript)
        else:
            transcription = transcript + overwrite_chars
            complete_transcription += transcription + " "
            st.write(transcription)  # Print the final transcription

            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                stream.stop()  # Stop the stream immediately
                # st.write('Exiting...')
                break

            num_chars_printed = 0
    return complete_transcription




import threading

import multiprocessing



def transcription_thread(client, streaming_config, stream, complete_transcription, stop_flag):
    audio_generator = stream.generator()
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)
    for response in responses:
        num_chars_printed = 0
        # if stream.stop_flag:
        if stop_flag.is_set():
            break
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        # complete_transcription.append(transcript)

        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            num_chars_printed = len(transcript)
        else:
            transcription = transcript + overwrite_chars
            complete_transcription += transcription + " "
            # st.write(transcription)  # Print the final transcription

            
        if re.search(r'\b(exit|quit)\b', transcript, re.I):
            stream.stop()
            stop_flag.set()  # Set the stop_flag to signal other threads to stop
            break

    return complete_transcription



def continuous_transcription_and_face_emotion():
    rate = 16000
    chunk = int(rate / 10)
    manager = multiprocessing.Manager()
    all_face_emotions = manager.list()
    complete_transcription = manager.list()
    stop_flag = threading.Event()

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code="en-US"
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    # model_path = 'facial_emotions_image_detection/checkpoint-3136'
    # pipe = pipeline('image-classification', model=model_path, device=-1)
    # face_model = YOLO('yolov8l-face.pt')


    video = cv2.VideoCapture(0)
    frame_placeholder = st.empty()


    with MicrophoneStream(rate, chunk) as stream:
        # face_thread = threading.Thread(target=face_analysis_thread, args=(video, pipe, face_model, frame_placeholder, all_face_emotions, stop_flag))
        audio_thread = threading.Thread(target=transcription_thread, args=(client, streaming_config, stream, complete_transcription, stop_flag))

        # face_thread.start()
        audio_thread.start()


        while not stop_flag.is_set():
            ret, frame = video.read()
            if not ret:
                break
            face_emotions = analyze_facial_emotion(frame, pipe, face_model)
            all_face_emotions.extend(face_emotions)
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)




        # face_thread.join()
        audio_thread.join()

        complete_transcription_str = "".join(complete_transcription)
        with wave.open(stream.output_filename, 'rb') as wave_file:
            frames = wave_file.readframes(wave_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
        
        if wave_file.getframerate() != 16000:
            audio_data = librosa.resample(audio_data.astype(float), orig_sr=wave_file.getframerate(), target_sr=16000)
            
        audio_data = audio_data.astype(np.float32)

    video.release()
    cv2.destroyAllWindows()

    face_emotions_series = pd.Series(list(all_face_emotions), name='Face Emotions')
    st.session_state.emotions_data = pd.DataFrame(face_emotions_series, columns=['Face Emotions'])

    return audio_data, complete_transcription_str, all_face_emotions




def split_text_into_segments(text, num_segments):
    segment_length = len(text) // num_segments
    segments = []
    last_index = 0

    for _ in range(num_segments - 1):
        split_index = text.rfind(' ', last_index, last_index + segment_length + 1)
        if split_index == -1:
            split_index = last_index + segment_length
        segments.append(text[last_index:split_index])
        last_index = split_index + 1

    segments.append(text[last_index:])
    return segments

def process_audio_chunk(audio, text):
    sample_rate = 16000
    chunk_length = 10 * sample_rate
    ignore_length = 2 * sample_rate

    all_audio_emotions = []
    all_text_emotions = []
    all_scores = []

    if 'Voice Emotions' in st.session_state.emotions_data.columns:
        st.session_state.emotions_data.drop(columns=['Voice Emotions'], inplace=True)
    if 'Text Emotions' in st.session_state.emotions_data.columns:
        st.session_state.emotions_data.drop(columns=['Text Emotions'], inplace=True)

    for start in range(0, len(audio) - ignore_length, chunk_length):
        end = start + chunk_length
        audio_chunk = audio[start:end]
        if len(audio_chunk) == 0:
            continue
        if is_silent(audio_chunk):
            continue

        audio_chunk = torch.from_numpy(audio_chunk).to(torch.float64).to(device)
        inputs = emotion_feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).squeeze()
            score_dict = {emotion_model.config.id2label[i]: score.item() for i, score in enumerate(scores)}
            all_scores.append(score_dict)

            initial_prediction = max(score_dict, key=score_dict.get)

            if initial_prediction == 'sad':
                adjusted_sad_score = score_dict['sad'] * 0.1
                adjusted_neutral_score = score_dict['neutral'] * 10

                if adjusted_neutral_score > adjusted_sad_score:
                    predicted_emotion = 'neutral'
                else:
                    predicted_emotion = 'sad'
            else:
                predicted_emotion = initial_prediction
            
            all_audio_emotions.append(predicted_emotion)

    voice_emotions_series = pd.Series(all_audio_emotions, name='Voice Emotions')

    text_chunks = split_text_into_segments(text, 4)
    for chunk in text_chunks:
        chunk = chunk.replace('exit', '')
        if len(chunk) == 0 or 'exit' in chunk.lower():
            continue

        text_inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            text_outputs = text_emotion_model(**text_inputs)
        text_prediction = torch.argmax(text_outputs.logits, dim=-1)
        text_emotion = text_emotion_model.config.id2label[text_prediction.item()]
        text_emotion = text_emotion_remap.get(text_emotion, text_emotion)
        
        all_text_emotions.append(text_emotion)

    text_emotions_series = pd.Series(all_text_emotions, name='Text Emotions')

    st.session_state.emotions_data = pd.concat([st.session_state.emotions_data, voice_emotions_series, text_emotions_series], axis=1)

    st.session_state.audio_emotions_df = pd.DataFrame({
        'Voice Emotions': all_audio_emotions
    })

    st.session_state.text_emotions_df = pd.DataFrame({
        'Text Emotions': all_text_emotions
    })

    return all_audio_emotions, all_text_emotions, st.session_state.audio_emotions_df, st.session_state.text_emotions_df, st.session_state.emotions_data

######### only audio ############
def continuous_transcription():
    rate = 16000
    chunk = int(rate / 10)

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    complete_transcription = ""

    with MicrophoneStream(rate, chunk) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if stream.stop_flag:
                break
            num_chars_printed = 0
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                # Clear the current line and print the interim result
                # st.write(transcript + overwrite_chars + '\r', end='', flush=True)
                num_chars_printed = len(transcript)
            else:
                transcription = transcript + overwrite_chars
                complete_transcription += transcription + " "
                # st.write(transcription)  # Print the final transcription
                st.write(transcript)

                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                # unique_key = str(uuid.uuid4())  # Generate a unique key for each button
                # if st.button("Stop", key=unique_key):
                    stream.stop()  # Stop the stream immediately
                    st.write('Exiting...')
                    break

            if stream.stop_flag:
                break

        with wave.open(stream.output_filename, 'rb') as wave_file:
            frames = wave_file.readframes(wave_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
        
        if wave_file.getframerate() != 16000:
            audio_data = librosa.resample(audio_data.astype(float), orig_sr=wave_file.getframerate(), target_sr=16000)
            
        audio_data = audio_data.astype(np.float32)

    
    return audio_data, complete_transcription

def process_audio_chunk_only(audio, text):
    sample_rate = 16000
    chunk_length = 10 * sample_rate
    ignore_length = 2 * sample_rate

    all_audio_emotions = []
    all_text_emotions = []
    all_scores = []

    if 'Voice Emotions' in st.session_state.emotions_data.columns:
        st.session_state.emotions_data.drop(columns=['Voice Emotions'], inplace=True)
    if 'Text Emotions' in st.session_state.emotions_data.columns:
        st.session_state.emotions_data.drop(columns=['Text Emotions'], inplace=True)

    for start in range(0, len(audio) - ignore_length, chunk_length):
        end = start + chunk_length
        audio_chunk = audio[start:end]
        if len(audio_chunk) == 0:
            continue
        if is_silent(audio_chunk):
            continue

        audio_chunk = torch.from_numpy(audio_chunk).to(torch.float64).to(device)
        inputs = emotion_feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).squeeze()
            score_dict = {emotion_model.config.id2label[i]: score.item() for i, score in enumerate(scores)}
            all_scores.append(score_dict)

            initial_prediction = max(score_dict, key=score_dict.get)

            if initial_prediction == 'sad':
                adjusted_sad_score = score_dict['sad'] * 0.1
                adjusted_neutral_score = score_dict['neutral'] * 10

                if adjusted_neutral_score > adjusted_sad_score:
                    predicted_emotion = 'neutral'
                else:
                    predicted_emotion = 'sad'
            else:
                predicted_emotion = initial_prediction
            
            all_audio_emotions.append(predicted_emotion)

    voice_emotions_series = pd.Series(all_audio_emotions, name='Voice Emotions')

    text_chunks = split_text_into_segments(text, 4)
    for chunk in text_chunks:
        chunk = chunk.replace('exit', '')
        if len(chunk) == 0 or 'exit' in chunk.lower():
            continue

        text_inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            text_outputs = text_emotion_model(**text_inputs)
        text_prediction = torch.argmax(text_outputs.logits, dim=-1)
        text_emotion = text_emotion_model.config.id2label[text_prediction.item()]
        text_emotion = text_emotion_remap.get(text_emotion, text_emotion)
        
        all_text_emotions.append(text_emotion)

    text_emotions_series = pd.Series(all_text_emotions, name='Text Emotions')

    st.session_state.emotions_data = pd.concat([st.session_state.emotions_data, voice_emotions_series, text_emotions_series], axis=1)

    st.session_state.audio_emotions_df = pd.DataFrame({
        'Voice Emotions': all_audio_emotions
    })

    st.session_state.text_emotions_df = pd.DataFrame({
        'Text Emotions': all_text_emotions
    })

    return all_audio_emotions, all_text_emotions, st.session_state.audio_emotions_df, st.session_state.text_emotions_df, st.session_state.emotions_data




# Streamlit UI
if page == 'Emotion Detection':

    st.title("Emotion Detect App")

    st.markdown(
        "###### Start by pressing 'Face & Audio Emotion Detection' or 'Audio Emotion Detection' button. You can stop processing by saying the word 'exit'. Then navigate to the next page using the sidebar to access the dashboard for comprehensive insights."
    )

    # Divide the page into two columns
    col1, col2 = st.columns(2)

    # Section for Facial Emotion Recognition
    with col1:
        st.header("Face & Audio Emotion Detection")
        if st.button("Start Camera and Microphone"):
            st.session_state.emotions_data = pd.DataFrame()
            audio_data, transcription, face_emotion = continuous_transcription_and_face_emotion()
            st.write(f"###### Predicted Facial Emotion: {face_emotion}")
            st.write(f"###### Transcription: {transcription}")

            # Process and display audio and text emotions
            if audio_data is not None and transcription:
                predicted_emotion, text_emotions, audio_emotions_df, text_emotions_df, emotions_data = process_audio_chunk(
                    audio_data, transcription)

                st.write(f"###### Predicted Speech Emotion: {predicted_emotion}")
                st.write(f"###### Detected Text Emotions: {text_emotions}")

                # st.dataframe(audio_emotions_df)
                # st.dataframe(text_emotions_df)
                # st.dataframe(emotions_data)


    # Section for Facial Emotion Recognition
    with col2:
        st.header("Audio Emotion Detection")
        if st.button("Start Microphone"):
            st.session_state.emotions_data = pd.DataFrame()
            audio_data, transcription = continuous_transcription()
            # st.write(f"###### Predicted Facial Emotion: {face_emotion}")
            # st.write(f"###### Transcription: {transcription}")

            # Process and display audio and text emotions
            if audio_data is not None and transcription:
                predicted_emotion, text_emotions, audio_emotions_df, text_emotions_df, emotions_data = process_audio_chunk(
                    audio_data, transcription)

                st.write(f"###### Predicted Speech Emotion: {predicted_emotion}")
                st.write(f"###### Detected Text Emotions: {text_emotions}")



if page == 'Dashboard':

    # st.header("Emotion Dashboard")
    if st.session_state.emotions_data.empty:
        st.write("Kindly initiate the face and audio processing. Upon completion, please revisit this page to access your comprehensive dashboard.")
    else:
        # Calculate overall counts and percentages
        overall_counts = st.session_state.emotions_data.stack().value_counts()
        total_emotions = overall_counts.sum()
        percentages = (overall_counts / total_emotions * 100).round(2)

        # Color dictionary for emotions
        color_dict = {
            'happy': 'gold',          # Joyful, bright, positive
            'sad': 'lightblue',       # Calm, subdued, melancholic
            'angry': 'red',           # Intense, aggressive
            'neutral': 'grey',        # Balanced, neutral
            'surprise': 'green',     # Fresh, unexpected
            'fear': 'orange',         # Alert, cautious
            'disgust': 'purple',      # Deep, aversive
            'anticipation': 'brown',  # Earthy, expectant
            'love': 'pink',           # Affectionate, soft
            'optimism': 'teal',       # Positive, confident
            'pessimism': 'maroon',    # Dark, negative
            'trust': 'blue'           # Trustworthy, stable
        }


        # Display at the top of the page using containers and columns
        with st.container():
            # Calculate the number of columns needed
            num_emotions = len(overall_counts)
            cols = st.columns(num_emotions)

            # Create a box for each emotion
            for i, emotion in enumerate(overall_counts.index):
                with cols[i]:
                    # Get color for the current emotion from the dictionary
                    color = color_dict.get(emotion, 'grey')  # Default to grey if not found
                    count = overall_counts[emotion]
                    percentage = percentages[emotion]
                    # # Create the box with both count and percentage
                    st.markdown(f"""
                    <div style="background-color: {color}; border-radius: 5px; padding: 5px 10px; text-align: center; color: white; font-size: 14px; line-height: 0.5;">
                        <h4 style="margin: 0; font-size: 14px;">{emotion.capitalize()}</h4>
                        <h3 style="margin: 0; font-size: 16px;">{count}</h3>
                        <p style="margin: 0; font-size: 12px;">({percentage}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                                

        # Most Predominant Emotion
        if not st.session_state.emotions_data.empty:
            # Count all non-null emotions across all columns and get the most frequent
            all_emotions = st.session_state.emotions_data.melt()['value']
            predominant_emotion = all_emotions.mode().iloc[0] if not all_emotions.mode().empty else "No predominant emotion detected"
            emotion_color = color_dict.get(predominant_emotion, 'grey')  # Get the color for the predominant emotion

            st.markdown(f"""
            <div style="background-color: {emotion_color}; border-radius: 8px; padding: 20px; color: white; text-align: center; margin-top: 20px; margin-bottom: 30px;">
                <h3 style="margin: 0;">Most Predominant Emotion</h3>
                <h1 style="margin: 0;">{predominant_emotion.capitalize()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.subheader("Most Predominant Emotion")
            st.write("No emotions detected.")

        # Add space divider
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)


        # Aggregate emotions
        emotion_counts = st.session_state.emotions_data.apply(pd.Series.value_counts).fillna(0)

        # Create columns to display the bar charts side by side
        cols = st.columns(3)
        column_iterator = iter(cols)

        for emotion_type in ['Face Emotions', 'Voice Emotions', 'Text Emotions']:
            if emotion_type in emotion_counts.columns:
                current_column = next(column_iterator)
                with current_column:
                    data_to_plot = emotion_counts[emotion_type][emotion_counts[emotion_type] > 0]
                    if not data_to_plot.empty:
                        colors = [color_dict.get(emotion, 'grey') for emotion in data_to_plot.index]
                        fig = go.Figure(go.Bar(
                            x=data_to_plot.values,
                            y=data_to_plot.index,
                            marker_color=colors,
                            orientation='h'
                        ))
                        # Adjust layout settings for visibility
                        fig.update_layout(
                            title=f'{emotion_type} Frequency',
                            xaxis_title='Frequency',
                            yaxis_title='Emotions',
                            yaxis={'categoryorder': 'total ascending'},
                            height=300,
                            margin=dict(l=40, r=40, t=40, b=20)  # Left, Right, Top, Bottom margins
                        )
                        st.plotly_chart(fig, use_container_width=True)  # Use full width of the column
                    else:
                        st.write(f"No {emotion_type.lower()} emotions recorded.")

        # Add space divider
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

        # Divide the page into two columns
        col1, col2 = st.columns([3, 2])

        with col1:
            # Create a list of colors for the pie chart, matching the order of 'overall_counts.index'
            pie_colors = [color_dict[emotion] for emotion in overall_counts.index if emotion in color_dict]

            # Create the pie chart with the specific colors
            fig = go.Figure(data=[go.Pie(labels=overall_counts.index, values=overall_counts.values, marker_colors=pie_colors)])
            # fig.update_layout(title="Overall Emotions Distribution")
            # Update layout to make the pie chart smaller
            fig.update_layout(
                title="Overall Emotions Distribution",
                height=400,  # Set the height of the pie chart
                width=400,   # Set the width of the pie chart
                margin=dict(l=20, r=20, t=30, b=20)  # Adjust margins for better spacing
            )
            st.plotly_chart(fig)

        # Add some space between the columns
        st.markdown("<hr>", unsafe_allow_html=True)

        with col2:
            # Add a title above the DataFrame
            st.markdown("###### Detected Emotions from Three Different Tasks")
            st.dataframe(st.session_state.emotions_data)
