from transformers import (
    AutoConfig,
    BlenderbotSmallForConditionalGeneration
                          )
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from huggingface_hub import hf_hub_url, cached_download
from onnxruntime import (GraphOptimizationLevel,
                        InferenceSession,
                        SessionOptions)

from torch import from_numpy
from torch.nn import Module
from functools import reduce
from operator import iconcat

model_vocab_size=30000
model_card="remzicam/xs_blenderbot_onnx"
model_file_names=["blenderbot_small-90M-encoder-quantized.onnx",
                "blenderbot_small-90M-decoder-quantized.onnx",
                "blenderbot_small-90M-init-decoder-quantized.onnx"]

class BlenderEncoder(Module):
    def __init__(self, encoder_sess):
        super().__init__()
        self.encoder = encoder_sess
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_hidden_state = from_numpy(
            self.encoder.run(
                None,
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)


class BlenderDecoderInit(Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):

        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "encoder_attention_mask": encoder_attention_mask.cpu().numpy(),
                "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
            },
        )

        list_pkv = tuple(from_numpy(x) for x in decoder_outputs[1:])

        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return from_numpy(decoder_outputs[0]), out_past_key_values


class BlenderDecoder(Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_output, past_key_values):

        decoder_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "encoder_attention_mask": attention_mask.cpu().numpy(),
        }

        flat_past_key_values = reduce(iconcat, past_key_values, [])

        past_key_values = {
            f"pkv_{i}": pkv.cpu().numpy() for i, pkv in enumerate(flat_past_key_values)
        }

        decoder_outputs = self.decoder.run(None, {**decoder_inputs, **past_key_values})
        # converts each value of the list to tensor from numpy
        list_pkv = tuple(from_numpy(x) for x in decoder_outputs[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return from_numpy(decoder_outputs[0]), out_past_key_values


class OnnxBlender(BlenderbotSmallForConditionalGeneration):
    """creates a Blender model using onnx sessions (encode, decoder & init_decoder)"""

    def __init__(self, onnx_model_sessions):
        config = AutoConfig.from_pretrained("facebook/blenderbot_small-90M")
        config.vocab_size=model_vocab_size
        super().__init__(config)

        assert len(onnx_model_sessions) == 3, "all three models should be given"

        encoder_sess, decoder_sess, decoder_sess_init = onnx_model_sessions

        self.encoder = BlenderEncoder(encoder_sess)
        self.decoder = BlenderDecoder(decoder_sess)
        self.decoder_init = BlenderDecoderInit(decoder_sess_init)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        encoder_hidden_states = encoder_outputs[0]
       
        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

class ModelLoad:
    def __init__(self, model_card,file_names):
        self.model_card=model_card
        self.file_names=file_names

    def model_file_downloader(self,model_card,filename):
        config_file_url = hf_hub_url(model_card, filename)
        model_file = cached_download(config_file_url)
        return model_file

    def inference_session(self,file_name):
        model_file=self.model_file_downloader(self.model_card,file_name)
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        return InferenceSession(model_file,options=options)

    def __call__(self,model_config):
        model=model_config([*map(self.inference_session,
                                self.file_names)])
        return model

model_loader=ModelLoad(model_card,model_file_names)
blender_onnx_model=model_loader(OnnxBlender)
