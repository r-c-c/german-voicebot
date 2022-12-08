# german-voicebot
It is a Voice Chatbot Gradio app to practice German language skills.
Bot generates answer via blenderbot. I convert original model into ONNX, then quantized it.
Model size (168mb) becomes less than half of original model size (350mb). Also it provides 2.5x faster cpu inference.

App demo link: https://huggingface.co/spaces/remzicam/voicebot_german

Model link: https://huggingface.co/remzicam/xs_blenderbot_onnx

I use google for speech recognition (speech to text), text to speech, and language translation.

# credit
To create the model, I adopted codes from https://github.com/siddharth-sharma7/fast-Bart repository.
