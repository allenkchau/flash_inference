# Model Configuration Hyperparameters

This doc just explains some of the hyperparameters defined in `flash_inference/model/config.py`.

---

## `hidden_dim`

Size of the vector that represents a single token as it flows through a NN.

---

## `vocab_size`

Total # of discrete tokens that the LM knows how to process; determines the size of the input embedding table and final output projection.

---

## `intermediate_dim`

MLP expansion size.

---

## `n_layers`

The total number of stacked transformer blocks.

---

## `n_heads`

Number of attention heads (query heads in the case of GQA), basically how many ways the model can look at the input sequence at the same time.

For example:

- Head 1 might focus on subject-verb agreement.
- Head 2 might pay attention to long-distance relationships.

---

## `n_kv_heads`

In GQA, multiple query heads share the same key and value head.

For example, if `n_heads = 9` and `n_kv_heads = 3`, 3 query heads attend to a single KV head.

---

## `tie_word_embeddings`

If true, the NN reuses the same weight matrix for the input token embedding and the very last output LM head (linear projection), which has shape `(vocab_size, hidden_dim)`.

This just reduces the model memory footprint (larger models usually don't tie because the savings are somewhat insignificant).

---

## `max_seq_len`

Max # of tokens the model can process in a single sequence, including the input prompt and the generated output tokens combined.

---

## `rms_norm_eps`

Tiny constant added inside RMS Norm formula to prevent division by zero and maintain numerical stability.

---

## `rope_theta`

Hyperparameter for the RoPE embedding (see RoPE doc).

---

## `attn/mlp bias`

Do we train learnable bias vectors that are added after the matmuls?
