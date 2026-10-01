# Normalization

Normalization acts as a stabilizer for the model's data flow, forcing the data to fit into a standardized range. It prevents gradient vanishing or explosion as the data passes through many layers.

This is important since vanishing or exploding gradients prevents the model from learning effectively.

Also, when features are on different scales, the network would have to make a large update to one weight compared to another weight during gradient descent. The descent trajectory can oscillate back and forth along the smaller, taking longer to reach the minimum.

If the features are on the same scale, the loss landscape is more uniform like a bowl and we can reach the minimum a lot faster.

---

## Vanishing and Exploding Gradients

Let's go a bit deeper into how gradients can explode or vanish. In an unnormalized transformer we could get a mix of both.

### Vanishing Gradients

If gradients shrink as backprop moves through deeper layers, earlier layers will receive very small updates, meaning they never improve upon random weight initialization.

This happens locally at specific connections like softmax where a dominant logit pushes the function into its flat outer edges, giving a very very small gradient.

Sometimes we still get softmax saturation but we have residual connections to counter that and multiple heads running in parallel (multiple softmaxes, so the node doesn't just die).

### Why Residuals?

**Prevents Gradient Death:** They create an uninterrupted mathematical highway from the last layer to the first. If a layer’s Softmax saturates and its local gradient drops to zero, the backpropagating gradient simply flows around it via the highway, keeping the rest of the model learning.

**Enables Deep Architectures:** Without residuals, signals vanish or explode after just a few layers. By providing a clean pathway for information, residuals are the exact reason we can successfully train ultra-deep models with 80+ layers.

**Simplifies the Optimization Task:** Instead of forcing each layer to completely reinvent and redraw the data vector from scratch, residuals allow layers to focus on calculating tiny micro-adjustments (`Δ`). The layer only learns what to add to the existing input.

**Maintains Core Identity:** They ensure that the original context of the words (the input tokens) is safely carried through the network, preventing deeper layers from accidentally forgetting or scrambling the initial prompt.

---

### Exploding Gradients

Here we have massive erratic adjustments to network weights and instead of converging to a minimal loss, it bounces around wildly or can jump to infinity (NaN).

This happens at linear/MLP steps since raw activation values inside the tokens have ballooned, and this acts as a massive multiplier on the gradient during the backward pass.

> **Good weight initialization is also key to making sure gradients don't explode or vanish on initial steps.**

---

# BatchNorm vs. LayerNorm vs. RMSNorm

Before RMSNorm let's talk about BatchNorm and LayerNorm.

---

## BatchNorm

**BatchNorm →** normalizes the activations across the entire batch for each individual feature dimension.

- Instead of requiring all normalized values to have zero mean and a variance of 1, BatchNorm has learnable shift (to a different mean) and scale (to a different variance) parameters.
- Keeps a running count of the exponential moving average of the mean and variance during mini-batch training but does nothing with it; only uses these values for single samples during inference.

### Example

Our input has the shape `(B, T, C)` where:

- `B = 2` → 2 sentences
- `T = 3` → each sentence has 3 tokens
- `C = 4` → each token is represented by a vector of 4 numbers

We look at a single feature column at a time across the entire batch.

So if we look only at **feature 0** (first number in every single word vector), we are normalizing a bucket of `B × T = 6` values with its special calculated mean and variance.

> **Important: bucket size is `B × T`.**

The same thing would apply if we had `(B, C, H, W)` for images where we would look at a single channel column at a time with `B × H × W` normalized values in that bucket.

---

## LayerNorm

**LayerNorm →** instead of normalizing across the batch, LayerNorm normalizes across the hidden/channel dimension for each individual token independently.

- To be clear, we normalize across the features of a single independent "entity", which is a single token for text and the entire image for images.
- Used in LLMs because BatchNorm bucket depends on sequence length; for sentences with different # of words/tokens if we want to use BatchNorm we would have to pad with empty tokens and this pollutes the math.
- In language the position of a word matters but its abstract meaning matters more, which is what LayerNorm captures; BatchNorm forces every single token at every position in a batch to share the same scale baseline.
- Also during inference BatchNorm is useless since our batch size is 1 and sequence length is 1 since we generate a single token at a time.

Some problems arise from this: we can't calculate a meaningful mean or variance from a single number.

BatchNorm works for CNNs because of the historical running averages I mentioned above to use during inference, but for text, sentences are too context-dependent for us to keep track of historical averages.

For LayerNorm in text, bucket size is the channel dimension which is fixed so the size is stable during training and inference.

### Example

Using the same example as BatchNorm, LayerNorm only looks at **sentence 1, word 0**, which is a vector of 4 values, and uses that to compute mean and variance.

It will do this for every single one of the 6 words individually and never mixes numbers from sentence 1 with sentence 2.

> **Important: bucket size is `C`.**

For the image, we would have `C × H × W` = avg all 12 color pixel values in the single image.

---

# RMSNorm

Now, RMSNorm is similar to LayerNorm except for the fact that they tossed out the shift step.

Researchers found that the mean doesn't really stabilize the model and it's purely the scale step (divide by standard deviation) that keeps the magnitude of the activations bounded so they don't explode or saturate the softmax function.

### Key Differences

- No learnable bias is added at the end although we still have `gamma`; it's just pure raw scaling.
- It reduces the computational overhead of the normalization layer.
- LayerNorm needs a pass to sum up the numbers and then another pass to subtract the mean, square them, etc.
- This hurts performance when we are bandwidth-bound.
- RMSNorm requires only a single pass to square the numbers and sum them all at once.
- Also fewer parameters to learn because no bias.

### Formula

Look at the formula for variance and std; RMS(x) is the same formula with `mean = 0`.
