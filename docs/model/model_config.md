This doc just explains some of the hyperparameters defined in flash_inference/model/config.py.

hidden_dim -> size of the vector that represents a single token as it flows through a NN

vocab_size -> total # of discrete tokens that the LM knows how to process; determines size of input embedding table and final output projection

intermediate_dim -> mlp expansion size

n_layers -> the total number of stacked transformer blocks

n_heads -> number of attention heads, basically how many ways the model can look at the input sequence at the same time; head 1 might focus on subject-verb agreement
while head 2 might pay attention to long distance relationships

max_seq_len -> max # of tokens the model can process in a single sequence including the input prompt and the generated output tokens combined


