import os
import nltk
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from googletrans import Translator

# ✅ Download necessary NLTK package
nltk.download("punkt")

# ✅ Configure Gemini API
# ✅ Set your Gemini API key here
genai.configure(api_key="AIzaSyCPjir_CPuKB0DjnH4HMuydzj-dji_CSTw")  # ← Replace this with your actual Gemini API key# Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Load emotion detection model from Hugging Face
emotion_model_name = "nateraw/bert-base-uncased-emotion"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# 🧠 Emotion detection function
def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_class_id = int(torch.argmax(logits))
    emotion = emotion_model.config.id2label[predicted_class_id]
    return emotion.lower()

# 🪴 Helpful mental health tips
tips = {
    "depression": [
        "You're not alone. Reach out to a friend or a mental health professional.",
        "Try journaling your thoughts and practicing gratitude each day.",
    ],
    "anxiety": [
        "Try deep breathing: inhale for 4 seconds, hold for 4, exhale for 4.",
        "Progressive muscle relaxation can help ease tension.",
    ],
    "stress": [
        "Break tasks into smaller steps to make them manageable.",
        "Take short breaks and step away from work or screens.",
    ],
    "selfcare": [
        "Prioritize sleep. Aim for 7-8 hours each night.",
        "Eat balanced meals and stay hydrated.",
    ]
}

# 🔄 Per-user context
user_context = {}         # For conversation history
user_languages = {}       # For preferred language
translator = Translator() # Google Translate

# 💬 Gemini response generation
def gemini_response(message, user_id="default"):
    if user_id not in user_context:
        user_context[user_id] = []

    try:
        chat = model.start_chat(history=user_context[user_id])
        prompt = (
            f"Keep your reply conversational. Make it short where needed and give suggestions only when the user asks for it. "
            f"Be empathetic, supportive, and think with respect to Indian culture and emotions. User said: '{message}'"
        )
        response = chat.send_message(prompt)
        user_context[user_id].append({"role": "user", "parts": [message]})
        user_context[user_id].append({"role": "model", "parts": [response.text]})
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# 🌟 Main handler with emotion detection and optional translation
def get_response(message, user_id="default"):
    try:
        # Detect if user is asking to switch language
        lower_msg = message.lower()
        change_lang_phrases = ["i want to talk in", "let's talk in", "can we talk in"]
        lang_detected = None

        for phrase in change_lang_phrases:
            if phrase in lower_msg:
                lang_detected = detect(message)
                user_languages[user_id] = lang_detected
                break

        # Default to English
        target_lang = user_languages.get(user_id, "en")

        # Translate input to English for Gemini/emotion
        if target_lang != "en":
            translated_input = translator.translate(message, dest="en").text
        else:
            translated_input = message

        # Detect emotion and generate reply
        emotion = detect_emotion(translated_input)
        response = gemini_response(translated_input, user_id)

        if emotion in tips:
            response += f"\n\n💡 *Based on your emotion ({emotion}), here are some tips:*\n" + "\n".join(tips[emotion])

        # Translate back to user-preferred language (if changed)
        if target_lang != "en":
            translated_response = translator.translate(response, src="en", dest=target_lang).text
            return translated_response
        else:
            return response

    except Exception as e:
        return f"⚠️ Error: {e}"
