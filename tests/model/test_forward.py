import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from flash_inference.model.config import ModelConfig


# def test_forward():
#     prompt = "Hello, what is your name?"
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     model_id = "HuggingFaceTB/SmolLM-135M"

#     # get the model and tokenizer from HF
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     model = model.to(device)

#     # load our scratch model
#     # config = ModelConfig()

#     # tokenize the text
#     inputs = tokenizer(prompt)
#     print(inputs)

    # run HF model

    # run our model

    # compare the output logits
    # assert torch.allclose()
