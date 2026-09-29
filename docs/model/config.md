This doc just explains some of the hyperparameters defined in flash_inference/model/config.py.

hidden_dim -> size of the vector that represents a single token as it flows through a NN

vocab_size -> total # of discrete tokens that the LM knows how to process; determines size of input embedding table and final output projection

intermediate_dim -> mlp expansion size

n_layers -> the total number of stacked transformer blocks

n_heads -> number of attention heads (query heads in the case of GQA), basically how many ways the model can look at the input sequence at the same time; head 1 might focus on subject-verb agreement
while head 2 might pay attention to long distance relationships

n_kv_heads -> in GQA, multiple query heads share the same key and value head (ex: if n_heads=9 and n_kv_heads=3, 3 query heads attend to a single kv head)

tie_word_embeddings -> if true, the NN reuses the same weight matrix for the input token embeding and the very last output LM head (linear projection) which has shape (vocab_size, hidden_dim); this 
just reduces the model memory footprint (larger models usaully don't tie because the savings are somewhat insignificant)

max_seq_len -> max # of tokens the model can process in a single sequence including the input prompt and the generated output tokens combined

rms_norm_eps -> tiny constant added iinside rms norm formula to prevent division by zero and maintain numerical stability

rope_theta -> hyperparamter for the RoPE embedding (see RoPE doc)

attn/mlp bias -> do we train learnable bias vectors that are added after the matmuls?



