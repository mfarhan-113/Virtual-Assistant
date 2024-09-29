import pyttsx3
import speech_recognition as sr
import wikipedia
import webbrowser
import pywhatkit as wk
import cv2
import pyautogui
import time
import operator
import requests
import sys
import os
import warnings
import smtplib
import psutil
import shutil
import matplotlib.pyplot as plt
from googletrans import Translator
from datetime import datetime
from email.message import EmailMessage
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
from tensorflow import keras
import spacy
from textblob import TextBlob
import numpy as np
from tensorflow.keras.utils import to_categorical

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module=".*dense.*")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Function to speak text
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Greet the user
def wishMe():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("Aye Aye Captain. Ready to serve you")

# Function to recognize user's speech
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.5
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-pk')
        print(f"User said: {query}\n")
    except Exception as e:
        print("Say that again please...")
        return "None"
    return query

# Function to get system information
def get_system_info():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    system_info = f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%, Disk Usage: {disk_usage}%"
    return system_info

# Function to get translation using Google Translate
def get_translation(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    translated_text = translation.text
    print(translated_text)
    return translated_text

# Function to add reminder to the reminders file
def add_reminder(time, task):
    with open("reminders.txt", "a") as file:
        file.write(f"{time}: {task}\n")

# Function to read reminders from the reminders file
def read_reminders():
    reminders = []
    with open("reminders.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            time, task = line.strip().split(": ")
            reminders.append((time, task))
    return reminders

# Function to handle reminder task
def reminder_task(task):
    speak("Reminder: " + task)

# Function to get weather information for a city
def weather_info(city):
    url = f'https://www.weather-forecast.com/locations/{city}/forecasts/latest'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        forecast = soup.find(class_='b-forecast__table-description-content').text.strip()
        return forecast
    else:
        return "Sorry, weather information for this city is not available."

# Function for Named Entity Recognition (NER) using spaCy
def ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function for Sentiment Analysis using TextBlob
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return polarity

if __name__ == "__main__":
    wishMe()
    scheduler = BackgroundScheduler()
    # Add job to read reminders every minute
    scheduler.add_job(read_reminders, 'interval', minutes=1)
    # Start the scheduler
    scheduler.start()

    intent_data = [
        ("Yes, sir", "greeting"),
        ("I am your virtual assistant programmed to assist you with various tasks.", "self introduction"),
        ("I was created by Farhan and Moiz using the Python language in PyCharm software.", "creator"),
        ("What do you want me to search for?", "search"),
        ("What do you want me to search for?", "search"),
        ("Opening Google", "open google"),
        ("Opening Google", "open google"),
        ("What do you want me to search for on YouTube?", "search youtube"),
        ("What do you want me to search for on YouTube?", "search youtube"),
        ("Closing browser", "close browser"),
        ("Opening Paint", "open paint"),
        ("Closing Paint", "close paint"),
        ("Opening Notepad", "open notepad"),
        ("Alright then, I am switching off", "go to sleep"),
        ("Closing Notepad", "close notepad"),
        ("What time is it?", "time"),
        ("Playing some music", "play_music"),
        ("Playing a movie", "play movie"),
        ("Closing media player", "close media"),
        ("Shutting down the system", "shutdown system"),
        ("Restarting the system", "restart system"),
        ("Locking the system", "lock system"),
        ("Opening Camera", "open camera"),
        ("Taking a screenshot", "take screenshot"),
        ("Calculating result", "calculate"),
        ("What is my IP address?", "ip address"),
        ("Increasing volume", "volume up"),
        ("Decreasing volume", "volume down"),
        ("Muting/unmuting volume", "mute unmute"),
        ("Noted", "noted"),
        ("Opening Chrome", "open chrome"),
        ("Maximizing window", "maximize window"),
        ("What do you want to search for on Google?", "google search"),
        ("What do you want to search for on YouTube?", "youtube search"),
        ("Opening new window", "open new window"),
        ("Opening incognito window", "open incognito window"),
        ("Minimizing window", "minimize window"),
        ("Opening history", "open history"),
        ("Opening downloads", "open downloads"),
        ("Moving to previous tab", "previous tab"),
        ("Moving to next tab", "next tab"),
        ("Closing tab", "close tab"),
        ("Closing window", "close window"),
        ("Clearing browsing history", "clear browsing history"),
        ("Closing Chrome", "close chrome"),
        ("What is the weather like in {city}?", "weather"),
        ("Setting a reminder", "set reminder"),
        ("Showing reminders", "show reminders"),
        ("Sending email", "send email"),
        ("Showing system information", "system info"),
        ("Translating text", "translate"),
        # Add more examples...
    ]

    # Tokenize and vectorize the input data
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([text for text, intent in intent_data])
    X_train = tokenizer.texts_to_matrix([text for text, intent in intent_data], mode='binary')
    intent_labels = {intent: idx for idx, intent in enumerate(set(intent for _, intent in intent_data))}
    y_train = [intent_labels[intent] for _, intent in intent_data]

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=len(intent_labels))

    # Fit the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1, callbacks=[TrainingLogger()])


    # Plot training accuracy and loss
    plt.figure(figsize=(12, 4))
    # Define and compile the neural network model for Intent Recognition
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        keras.layers.Dense(len(intent_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Add a callback to log training metrics
    class TrainingLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1}: Loss = {logs['loss']}, Accuracy = {logs['accuracy']}")

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Main loop to listen and respond to user commands
    while True:
        query = takeCommand().lower()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")

        elif 'open google' in query:
            webbrowser.open("google.com")

        elif 'play' in query:
            song = query.replace('play', '')
            speak('playing ' + song)
            wk.playonyt(song)

        elif 'search' in query:
            query = query.replace("search", "")
            webbrowser.open(query)

        elif 'time' in query:
            strTime = datetime.now().strftime("%H:%M:%S")
            speak(f"Sir, the time is {strTime}")

        elif 'open code' in query:
            codePath = "C:\\path\\to\\your\\editor.exe"
            os.startfile(codePath)

        elif 'email to' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = "email@example.com"
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("I am not able to send this email")

        elif 'what is my ip address' in query:
            ip = requests.get('https://api.ipify.org').text
            speak(f"Your IP address is {ip}")

        elif 'translate' in query:
            speak("What text do you want to translate?")
            text_to_translate = takeCommand()
            translated_text = get_translation(text_to_translate)
            speak(translated_text)

        elif 'reminder' in query:
            if 'set' in query:
                speak("What should I remind you about?")
                task = takeCommand()
                speak("When should I remind you? Please specify the time in HH:MM format.")
                time = takeCommand()
                add_reminder(time, task)
                speak(f"Reminder set for {time} to {task}")
            elif 'show' in query:
                reminders = read_reminders()
                if reminders:
                    for time, task in reminders:
                        speak(f"Reminder at {time} to {task}")
                else:
                    speak("You have no reminders.")

        elif 'system info' in query:
            info = get_system_info()
            speak(info)

        elif 'weather' in query:
            speak("Which city's weather do you want to know?")
            city = takeCommand()
            forecast = weather_info(city)
            speak(f"The weather in {city} is {forecast}")

        elif 'stop' in query:
            speak("Goodbye!")
            break

