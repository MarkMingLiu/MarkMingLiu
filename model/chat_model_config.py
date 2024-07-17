from transformers import T5Config,LlamaConfig

class TextToTextModelConfig(T5Config):
    model_type = 't5'

class TextToTextLlamaModelConfig(LlamaConfig):
    model_type = 'llama'