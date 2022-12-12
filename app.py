from transformers import (BlenderbotSmallTokenizer,
                        logging)
from mtranslate import translate
from io import BytesIO
from base64 import b64encode
import gradio as gr
from speech_recognition import Recognizer,AudioFile
from gtts import gTTS
from blender_model import blender_onnx_model

#supress huggingface warnings
logging.set_verbosity_error()
bot_tokenizer_name="facebook/blenderbot_small-90M"
max_answer_length=100
bot_language="en"
main_language = 'de'
bot_tokenizer = BlenderbotSmallTokenizer.from_pretrained(bot_tokenizer_name)
#load chatbot model
bot_model=blender_onnx_model

def app(audio):
    """
    It takes voice input from user then 
    responds it both verbally and in text.
    """
    text=stt(audio)
    bot_response_en,bot_response_de=answer_generation(text)
    voice_bot=tts(bot_response_de)
    b64 = b64encode(voice_bot).decode()
    #html code that automatically play sounds
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return text,html,bot_response_de,bot_response_en

def stt(audio):
    """
    speech to text converter

    Args:
        audio: record of user speech

    Returns:
        text (str): recognized speech of user
    """
    r = Recognizer()
    # open the file
    with AudioFile(audio) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data,
                                    language=main_language)
    return text

def answer_generation(user_input_de:str):
    """
    it takes user input as text in german language. 
    Then it translates into English. Blenderbot works only in English.
    Then the model generates an answer w.r.t English version of the input.
    Finally, bot's response is translated into German.

    Args:
        user_input (str): text version of user's speech

    Returns:
        translated_bot_response (str): bot's response in german language
    """
    #de-en translation
    user_input_en=translate(user_input_de,
                            bot_language,
                            main_language)
    inputs = bot_tokenizer(user_input_en,
                            return_tensors="pt")
    generation= bot_model.generate(**inputs,
                            max_length=max_answer_length)
    bot_response_en=bot_tokenizer.decode(generation[0],
                            skip_special_tokens = True)
    #en-de translation
    bot_response_de=translate(bot_response_en,
                            main_language,
                            bot_language)
    
    return bot_response_en,bot_response_de

def tts(text:str):
    """converts text into audio bytes

    Args:
        text (str): generated answer of bot

    Returns:
        bytes_object(bytes): suitable format for html autoplay sound option
    """
    tts = gTTS(text=text,
                lang=main_language,
                slow=False)
    bytes_object = BytesIO()
    tts.write_to_fp(bytes_object)
    bytes_object.seek(0)
    return bytes_object.getvalue()

logo_image_path="German_AI_Voicebot.png"
logo = f"<center><img src='file/{logo_image_path}' width=180px></center>"
gr.Interface(
    fn=app, 
    inputs=[
        gr.Audio(source="microphone", type="filepath",
        ), 
    ],
    outputs=[
        gr.Textbox(label="You said: "),
        "html",
        gr.Textbox(label="AI said: "),
        gr.Textbox(label="AI said (English): "),
    ],
    live=True,
    allow_flagging="never",
    description=logo,
    ).launch()
