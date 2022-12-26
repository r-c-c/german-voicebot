---
license: other
---
# xs_blenderbot_onnx (only 168 mb)
onnx quantized version of facebook/blenderbot_small-90M model (350 mb)

Faster cpu inference

## INTRO

Before usage:

  • download blender_model.py script from files in this repo

  • pip install onnxruntime

you can use the model with huggingface generate function with its all parameters

# Usage

With text generation pipeline

```python
>>>from blender_model import TextGenerationPipeline

>>>max_answer_length = 100
>>>response_generator_pipe = TextGenerationPipeline(max_length=max_answer_length)
>>>utterance = "Hello, how are you?"
>>>response_generator_pipe(utterance)
i am well. how are you? what do you like to do in your free time?
```
Or you can call the model

```python
>>>from blender_model import OnnxBlender
>>>from transformers import BlenderbotSmallTokenizer
>>>original_repo_id = "facebook/blenderbot_small-90M"
>>>repo_id = "remzicam/xs_blenderbot_onnx"
>>>model_file_names = [
    "blenderbot_small-90M-encoder-quantized.onnx",
    "blenderbot_small-90M-decoder-quantized.onnx",
    "blenderbot_small-90M-init-decoder-quantized.onnx",
]
>>>model=OnnxBlender(original_repo_id, repo_id, model_file_names)
>>>utterance = "Hello, how are you?"
>>>inputs = tokenizer(utterance,
                    return_tensors="pt")
>>>outputs= model.generate(**inputs,
                        max_length=max_answer_length)
>>>response = tokenizer.decode(outputs[0],
                        skip_special_tokens = True)
>>>print(response)
i am well. how are you? what do you like to do in your free time?
```

# Credits
To create the model, I adopted codes from https://github.com/siddharth-sharma7/fast-Bart repository.
