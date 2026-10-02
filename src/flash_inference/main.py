import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from flash_inference.model.config import ModelConfig
from flash_inference.model.transformer import Transformer


def main():
    prompt = "Hello, what is your name?"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "HuggingFaceTB/SmolLM-135M"

    # get the model and tokenizer from HF
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_model = hf_model.to(device)

    # load our scratch model and initialize it with weights from HF
    config = ModelConfig()
    model = Transformer(config)



    # tokenize the text
    inputs = tokenizer(prompt)
    print(inputs)

    # pass the inputs to the models
    hf_model(inputs)
    model(inputs)


if __name__ == "__main__":
    main()
