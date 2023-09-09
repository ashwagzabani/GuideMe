# # # from django.http import HttpResponse
# # # from django.views.decorators import gzip
# # # from django.http import StreamingHttpResponse
# # # import cv2
# # # import threading


# # # def index(request):
# # #     return HttpResponse("Hello, world. You're at the polls index.")

# # # class VideoCamera(object):
# # #     def __init__(self):
# # #         self.video = cv2.VideoCapture(0)
# # #         (self.grabbed, self.frame) = self.video.read()
# # #         threading.Thread(target=self.update, args=()).start()

# # #     def __del__(self):
# # #         self.video.release()

# # #     def get_frame(self):
# # #         image = self.frame
# # #         _, jpeg = cv2.imencode('.jpg', image)
# # #         return jpeg.tobytes()

# # #     def update(self):
# # #         while True:
# # #             (self.grabbed, self.frame) = self.video.read()

# # #     def gen(camera):
# # #     while True:
# # #         frame = camera.get_frame()
# # #         yield(b'--frame\r\n'
# # #               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# # # @gzip.gzip_page
# # # def livefe(request):
# # #     try:
# # #         cam = VideoCamera()
# # #         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
# # #     except:  # This is bad! replace it with proper handling
# # #         pass
# # # import cv2
# # # import numpy as np
# # # import tensorflow as tf
# # # from django.shortcuts import render
# # # from django.http import HttpResponse
# # # from django.views.decorators.http import require_GET

# # # def detect_object(request):
# # #     # Open the live webcam
# # #     camera = cv2.VideoCapture(0)

# # #     # Touch anywhere in the screen to start listening to user prompt
# # #     while True:
# # #         ret, frame = camera.read()
# # #         cv2.imshow("Live Webcam", frame)
# # #         key = cv2.waitKey(1) & 0xFF

# # #         # If the user touches anywhere in the screen, start listening to the user prompt
# # #         if key == 27:
# # #             break

# # #     # Get the user prompt
# # #     user_prompt = input("What object do you want to detect? ")

# # #     # Load the TensorFlow object detection model
# # #     model = tf.saved_model.load("./object_detection_model")

# # #     # Detect the object in the live webcam
# # #     detections = model.detect(frame)

# # #     # If the object is detected, speech the label
# # #     if len(detections) > 0:
# # #         object_name = detections[0].get("name")
# # #         cv2.putText(frame, object_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# # #         cv2.imshow("Live Webcam", frame)
# # #         cv2.waitKey(0)

# # #         # Speech the object label
# # #         print("The object is: " + object_name)

# # #     else:
# # #         print("Object not detected")

# # #     return HttpResponse("Object detection complete")

# # # def index(request):
# # #     # Render the HTML template
# # #     return render(request, "index.html")

# # import cv2
# # from django.http import HttpResponse

# # def index(request):
# #     # Create a VideoCapture object
# #     cap = cv2.VideoCapture(0)

# #     # Start a while loop to capture frames from the webcam
# #     while True:
# #         # Capture the frame from the webcam
# #         ret, frame = cap.read()

# #         # Convert the frame to RGB
# #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #         # Encode the frame as JPEG
# #         jpeg_frame = cv2.imencode('.jpg', rgb_frame)[1].tobytes()

# #         # Create a response object
# #         response = HttpResponse(content=jpeg_frame, content_type='image/jpeg')

# #         # Return the response object
# #         return response

# import cv2
# from django.http import HttpResponse
# from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
# def index(request):
#     # Get the webcam feed
#     webcam = cv2.VideoCapture(0)

#     # Create a frame buffer
#     frame = None

#     # Start a loop to continuously read frames from the webcam
#     while True:
#         # Read the next frame from the webcam
#         success, frame = webcam.read()

#         # If the frame was not read successfully, break the loop
#         if not success:
#             break

#         # Convert the frame to a JPEG image
#         jpeg_image = cv2.imencode('.jpg', frame)[1].tobytes()

#         # Create a response object
#         response = HttpResponse(content=jpeg_image, content_type='image/jpeg')

#         # Return the response object
#         return response

# --------------------------------------
# from django.views.decorators import gzip
# from django.http import StreamingHttpResponse
# import cv2
# import threading

# class VideoCamera(object):
    
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         # cv2.imshow('mouseRGB', self.frame)
#         threading.Thread(target=self.update, args=()).start()
        
#         # cv2.setMouseCallback("new window",onclick)

#         # cv2.imshow('frame', self.video)

#         # cv2.namedWindow("frame") 
#         # cv2.setMouseCallback("frame", onclick)

#     def __del__(self):
#         # cv2.namedWindow("frame") 
#         # cv2.setMouseCallback("frame", onclick)
#         self.video.release()
#         # cv2.destroyAllWindows()
        
       

#     def get_frame(self):
#         image = self.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
            
#             # cv2.namedWindow('mouseRGB')
#             # cv2.setMouseCallback('mouseRGB',onclick)
#             # cv2.imshow('frame', self.video)
#             # cv2.imshow("GFG",self.video)
#             # cv2.setMouseCallback('frame', onclick,param=self.frame)
#             # if cv2.waitKey(25) & 0xFF == ord('q'):
#             #     break
        
            

            

# def onclick(event, x,y, flags, param):
#     print("jjjj")
    
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         cv2.imshow("frame",frame)
#         cv2.setMouseCallback('frame', onclick)
#         # if cv2.waitKey(25) & 0xFF == ord('q'):
#         #     break
#         # cv2.namedWindow('new window')
#         # cv2.imshow('new window', frame)
        


# @gzip.gzip_page
# def index(request):
#     try:
#         cam = VideoCamera()
        
#         # cv2.setMouseCallback('new window', function1)
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame",streaming=True)
#     except:  # This is bad! replace it with proper handling
#         pass

# ==============
import cv2
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
# import pyttsx3
# from gTTS import say
# from gtts import * as ddds
from gtts import gTTS
import os
from io import BytesIO
from playsound import playsound

# import sounddevice as sd
# import soundfile as sf
# from pydub import AudioSegment
# import pyaudio
# import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from soundfile import write

# speech
import whisper
import os
import numpy as np
import torch
import ssl
import json

# detect txt
# import pytesseract
import easyocr


def onclick(event, x,y, flags, param):
    print("jjjj")

@gzip.gzip_page
def lives(request):
    # Function to capture frames from webcam and stream it as HTTP response
    def generate_frames():

        # Function to capture frames from webcam
        cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras
        cv2.setMouseCallback("Frame", onclick)
        while True:
            ret, frame = cap.read()
            # cv2.imshow("frame",frame)
            
            if not ret:
                break
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            # Read the modified JSON data
            with open('config.json', 'r') as file:
                modified_data = json.load(file)
            if modified_data["detect_text"]:
                detect_text(frame)
            # Print the modified JSON data
            # print(modified_data)
            # ssl._create_default_https_context = ssl._create_unverified_context
            # reader = easyocr.Reader(['en'])
            # # Convert the frame to grayscale
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #         # Use EasyOCR to detect text from the frame
            # results = reader.readtext(gray)

            #         # Process the detected text
            # detected_text = []
            # for (bbox, text, _) in results:
            #     detected_text.append(text)

            # Print the detected text
            # if detected_text:
            #     print("detected_text",detected_text)
            # ssl._create_default_https_context = ssl._create_unverified_context
            # load model
            # reader = easyocr.Reader(['ch_tra', 'en'])

            # cv2.imshow("Frame",frame)
            # cv2.namedWindow("Frame")
            # key = cv2.waitKey(1)
            # if key == 27:
            #     break
            # cv2.setMouseCallback('GFG', mouse_click,param=frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            # if not ret:
            #     break
            # Yield the frame in bytes format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # cap.release()
        # cap.destroyAllWindows()
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



def index(request):
    # Function to handle mouse click events
    # if request.method == 'POST':
    #     x = int(request.POST.get('x'))
    #     y = int(request.POST.get('y'))
        # print(f'Mouse clicked at ({x}, {y})')
    # engine = pyttsx3.init()
    # engine.say("I will speak this text")
    # engine.runAndWait()
    # obj = say(language='en-us', text="Text From Views")
    # myobj = gTTS(text="mytext", lang="en", slow=False)
    # myobj.save("welcome.mp3")
    # os.system("welcome.mp3")
    # text_to_speech()
    # routput_file = 'recorded_audio.mp3'
    # duration = 5  # Duration in seconds

    # record_audio(output_file, duration)
    # t()
    speech_reconition()
    return render(request, 'index.html')

def text_to_speech(request):
    lang='en'
    text = "How can I help you?"
    tts = gTTS(text=text, lang=lang)
    tts.save('output.mp3')
    playsound('output.mp3')
    recording()
    return HttpResponse('Function executed successfully!')

def tts(content):
    lang='en'
    text = content
    tts = gTTS(text=text, lang=lang)
    tts.save('output.mp3')
    playsound('output.mp3')

# def record():
#     # Set the desired audio parameters
#     sample_rate = 44100  # Sample rate of the audio (in Hz)
#     duration = 10  # Duration of the recording (in seconds)
#     output_file = "file.mp3"  # Path to the output MP3 file
#     # Create an empty list to store the recorded audio frames
#     frames = []

#     # Define a callback function to capture the audio frames
#     def callback(indata, frames, time, status):
#         frames.extend(indata)

#     # Start recording the audio
#     with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
#         print(f"Recording started. Listening for {duration} seconds...")
#         sd.sleep(int(duration * 1000))

#     # Convert the recorded frames to bytes
#     audio_data = b''.join(frames)
#     # Save the recorded audio as WAV file
#     output_wav = "path/to/output/file.wav"  # Path to the output WAV file
#     sf.write(output_wav, audio_data, sample_rate)

#     # Load the recorded audio from the WAV file
#     audio = AudioSegment.from_wav(output_wav)

#     # Export the audio to MP3 format
#     audio.export(output_file, format="mp3")

#     print(f"Recording saved as {output_file}")

# def record_audio(output_file, duration):
#     # Set the audio parameters
#     format = pyaudio.paInt16
#     channels = 1
#     sample_rate = 44100
#     chunk_size = 1024

#     # Create an instance of the PyAudio class
#     audio = pyaudio.PyAudio()

#     # Open the audio stream
#     stream = audio.open(format=format,
#                         channels=channels,
#                         rate=sample_rate,
#                         input=True,
#                         frames_per_buffer=chunk_size)

#     print('Recording started...')

#     # Initialize an empty list to store the audio frames
#     frames = []

#     # Record audio for the specified duration
#     for i in range(0, int(sample_rate / chunk_size * duration)):
#         data = stream.read(chunk_size)
#         frames.append(data)

#     print('Recording finished.')

#     # Stop and close the audio stream
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     # Save the recorded audio to an MP3 file
#     sf.write(output_file, b''.join(frames), sample_rate)

#     print(f'Saved audio to {output_file}.')

def recording():
    # # Sampling frequency
    # freq = 44100

    # # Recording duration
    duration = 5
    # # Start recorder with the given values of 
    # # duration and sample frequency
    # recording = sd.rec(int(duration * freq), 
    #                 samplerate=freq, channels=2)

    # # Record audio for the given number of seconds
    # sd.wait()
    # # This will convert the NumPy array to an audio
    # # file with the given sampling frequency
    # write("recording0.wav", freq, recording)
    # wv.write("recording1.wav", recording, freq, sampwidth=2)
    # Sampling frequency
    freq = 44100

    # Start recorder with the given values of 
    # duration and sample frequency
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()
    output_file = 'recorded_audio.wav'
    # Save the recorded audio as WAV file
    write(output_file, recording, freq)
    speech_reconition()

    print(f'Saved audio to {output_file}.')

def speech_reconition():
    # ssl._create_default_https_context = ssl._create_unverified_context

    # model = whisper.load_model("base")
    # result = model.transcribe("welcome.mp3")
    # print(result["text"])
    # torch.cuda.is_available()
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # model = whisper.load_model("base", device=DEVICE)
    # print(
    #     f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    #     f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    # )
    # audio = whisper.load_audio("welcome.mp3")
    # audio = whisper.pad_or_trim(audio)
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")
    # # options = whisper.DecodingOptions(language=&quot;en&quot;, without_timestamps=True, fp16 = False)
    # result = whisper.decode(model, mel)
    # print(result.text)
    # result = model.transcribe("../input/audiofile/audio.mp3")
    # print(result["text"])
    model = whisper.load_model("base")
    result = model.transcribe("recorded_audio.wav")
    print("ddd",result["text"])
    if "speech" in result["text"] or "read" in result["text"] or "text" in result["text"] or "sentance" in result["text"]:
        with open('config.json', 'r') as file:
            json_data = json.load(file)

        # Modify the value in the JSON data
        json_data['detect_text'] = True
        # Write the modified JSON data back to the file
        with open('config.json', 'w') as file:
            json.dump(json_data, file, indent=4)
    print("")

# def detect_text():
#     # Initialize the EasyOCR reader
#     ssl._create_default_https_context = ssl._create_unverified_context
#     reader = easyocr.Reader(['en'])
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Use EasyOCR to detect text from the frame
#     results = reader.readtext(gray)

#             # Process the detected text
#     detected_text = []
#     for (bbox, text, _) in results:
#         detected_text.append(text)

#             # Print the detected text
#     if detected_text:
#         print("detected_text",detected_text)
#             # Display the frame
#             # cv2.imshow('Text Detection', frame)

#             # Break the loop when 'q' is pressed
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#     print("done")

# Function to handle the text detection
def detect_text(frame):
    ssl._create_default_https_context = ssl._create_unverified_context
    reader = easyocr.Reader(['en'])
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to detect text from the frame
    results = reader.readtext(gray)

    # Process the detected text
    detected_text = []
    sentance = ""
    for (bbox, text, _) in results:
        detected_text.append(text)
        tts(text)
    
    if len(detected_text) >0:
        for text in detected_text:
            sentance += " " + text

    # Print the detected text
    if detected_text:
        print("detected_text", detected_text)
        print("sentance", sentance)
        # Read the JSON file
        with open('config.json', 'r') as file:
            json_data = json.load(file)

        # Modify the value in the JSON data
        json_data['detect_text'] = False
        # Write the modified JSON data back to the file
        with open('config.json', 'w') as file:
            json.dump(json_data, file, indent=4)
        

        




