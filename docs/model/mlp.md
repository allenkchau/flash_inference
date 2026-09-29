# MLP (Multi-Layer Perceptron)

This is the component of the transformer block that processes individual token representations after the attention mechanism has gathered context. MLPs hold roughly **2/3 of all parameters** in a standard transformer.

At this point we now have a matrix where each row is a single token (the vector representation of that token).

---

## Normal MLP Architecture

This is what the architecture of a normal MLP would look like:

1. **Expansion:** Multiply the input by the first weight matrix which usually has dimensions `hidden_dim × intermediate_dim`. This is an expansion step.

2. **Nonlinear activation:** Apply a nonlinear activation function to the resulting matrix. This adds "thinking" capability.

3. **Projection back down:** Multiply step 2 result with a second weight matrix (`intermediate_dim × hidden_dim`) that projects back down so each row is once again a token vector with the original `hidden_dimension`.

---

## SwiGLU

But SmolLM uses the **SwiGLU activation**. SwiGLU has greater mathematical expressivity than ReLU or GeLU.

It forks the token processing into a **two-track parallel highway**.

The input token vector is fed into **2 separate linear projections** generating 2 independent representations in `intermediate_dim` space.

### 1. Gate Path

> Learn when and where to apply filters

Basically which parts of an input vector are important and what is noise for a given context.

### 2. Up Path

> Learn actual feature values and factual patterns

The output of the gate path is passed through the **SiLU activation function**, which is differentiable everywhere.

Then the activated gate path and linear up path do an **element-wise multiplication**.

This selectively "wipes out" data or lets it pass through (hence the name gate or filter).

Finally we do a **down projection** back to the model's original `hidden_dim`.

---

## Why No Bias?

The linear layers in the MLP usually do not have a learned bias vector for 3 primary reasons:

1. **Smaller memory footprint and more computationally efficient with kernels**
2. **Bias terms can introduce subtle training instability** (RMSNorm doesn't center the activations)
3. **Empirically, leaving them out doesn't harm model capacity**
