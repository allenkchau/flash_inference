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

    # create config and load our scratch model with weights from HF
    # config = ModelConfig.from_huggingface(model_id)
    # model = Transformer(config).to(device)

    # tokenize the text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(inputs)

    # pass the inputs to the models
    # HF does greedy decoding by default
    hf_outputs = hf_model.generate(
        inputs["input_ids"],
        max_length=50,
    )

    # outputs = model(inputs)

    print(hf_outputs)


if __name__ == "__main__":
    main()
