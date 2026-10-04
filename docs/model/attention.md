I will first explain the traditional multi-head attention and then expand a bit more on the grouped query attention variant that is used in SmolLM.

Say we have a sentence "The animal didn't cross the street because it was too tired".

At a high level self-attention allows the model to do things like associate "it" with "animal". In other words, we can look at other positions in the input sequence
for clues that can help lead to a better encoding for the word we are looking at.

Honestly to get a good understanding I would just read The Illustrated Transformer article by Jay Alammar.

query -> What am I looking for? This vector represents the current word that is actively searching the rest of the text for context.
key -> What do I offer? This vector acts like an indexing label or a tag for every word in the sequence, describing what kind of information that word contains.
value -> What content do I actually hold? Once the system matches the query with the keys, this vector contains the actual content or meaning that gets extracted and passed along.

In attention when we say we have multiple heads, that just means we have multiple sets of Q/K/V randomly initialized weight matrices. So we get n_head outputs of the attention operation.
Each head is used to project the input embeddings into a different representation subspace.

So the way to get back to a single matrix is to concatenate the outputs from the heads and then multiply by a final weight matrix Wo and that output is sent to the FFN.

But implementation wise we do something different. Instead of having n_head matrices, we combine them into a single matrix of size n_heads x head_dim which means we compute the query vectors
for all heads at the same time.

We do have an attention mask. This is normally mainly used during training but for the inference engine, the mask is activated during prefill. 
When the engine processes the input prompt, a token at a certain position should only be able to attend to itself and the tokens before it. 
If we didn't have the mask for prefill, we create a train-test mismatch (data the model sees during training comes from a much different distribution/environment than the data
it sees during testing/deployment) which means the activations will be garbage, etc.

When we do Q @ K.T we get a prompt_len by prompt_len grid of attention/compatibility scores where every row represents a query (token asking a question) and every column represents a key (token providing context).
The tokens in row 1 should only see columns 0 and 1. Columns 2, 3... are the tokens in the future.

So back to the mask. After Q @ K.T we add an upper triangular matrix of -infs. So now the upper triangular positions of the compatibility scores is -infs and then when we run sfotmax, the model zeros out
the attention of that query with regards to future tokens. So for the case of row 1, the attention weight score in columns 2, 3... are all 0 and the model splits 100% of its attention between the first
and second tokens.

We don't have to worry about the mask during decode because if we have a KV cache, we just feed the newly generated token into the model and get a vector of attention scores (for every token in the prompt
and the ones generated so far). There are no future tokens to worry about or mask out because they don't exist yet.

GQ attention is a bit different. 

Motivation: During inference, the KV cache becomes very large. When generating a token at a time, the model stores the Keys and Values of previous tokens in memory.

Each individual head has its own dedicated KV cache. In GQA, instead of every head having its own cache, a group of multiple query heads shares a single KV cache slot.
ex: If we have 8 heads in MHA, we have 8 Query heads, 8 Key heads etc. -> 8 distinct sets of KV vectors in memory
ex: If we have 8 query heads in GQA, we could slice them into 2 groups so this means we only store 2 key heads and 2 value heads in our cache

Multi-Query attention (MQA) is another variation where all query heads share a single KV cache. This results in the most memory savings but at the cost of degraded performance since less heads
means less ways the model can interpret a token simultaneously. Less nuance.

Remember each attn head projects the same token into a different vector space (same vector dimesnions but very different learned weights) allowing them to focus on different "meanings" at the same time.


