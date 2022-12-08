# german-voicebot
It is a Voice Chatbot app to practice German language skills.

It takes voice input in german from user.

Bot generates response via blenderbot as text.

After that, app sends the audio version of text in german. 

I convert original model into ONNX, then quantized it.

Model size (168mb) becomes less than half of original model size (350mb). 

It provides 2.5x faster cpu inference.

Deployed via Gradio.

App demo link: https://huggingface.co/spaces/remzicam/voicebot_german

App video link: https://www.youtube.com/watch?v=_xfTRkh47TY

Model link: https://huggingface.co/remzicam/xs_blenderbot_onnx

I use google API based libraries for speech recognition (speech to text), text to speech, and language translation.

# credit
To create the model, I adopted codes from https://github.com/siddharth-sharma7/fast-Bart repository.
