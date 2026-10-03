Let's start from the beginning.

Before we our input enters the actual transformer model, it gets embedded into 

The final transformer looks something like this:

- first we embedd
2. n_layer layers of transformer blocks
3. An LM head output matrix that returns the logits 

For every token you feed in the trasnformer, we get exactly one token output.
This is how the transformer is trained. If you feed a 2048 token document into the model, we get 2048 rows of logits simultaneously and we can calculate the loss for 2048
next word predictions in parallel and do the weight updates in the backward pass.

During inference we only care about the token we just generated, we don't care about the rows of the logits for the tokens before because we already know which tokens
are the ones that come before. We get logits of shape (batch_size, seq_len, hidden_dim) and we just care about logits[:, -1, :] since that will give us the next token.

