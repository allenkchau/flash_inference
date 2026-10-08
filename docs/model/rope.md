# Rotary Positional Embeddings (RoPE)

## Why Do We Need Positional Information?

In standard attention, we project each word into a Q and K and compute the dot product. It's just multiplying numbers and adding them up.

A dot product measures semantic similarity, not spatial order.

So this means for these sentences:

1. "Dog bites man"
2. "Man bites dog"

If there is no causal masking and no positional embeddings (like BERT-style attention) → both sentences are just an unordered bag of words to the model. It doesn't know who is biting who.

---

## Causal Masking Helps, But Isn't Enough

Let's look at a longer sentence:

> "The dog that chased the three cats across the busy street barked."

If there is causal masking (GPT) but no positional embeddings, words can only look backward, but it still can't tell whether a past token was 1 step away or 49 steps away.

- When computing attention for "barked", both "dog" and "cats" are valid nouns in the past.
- Without positional awareness, we can't distinguish the main subject.
- The model also can't naturally learn syntactic rules like adjectives usually modifying the noun immediately adjacent to them.

---

# Absolute Positional Embeddings

In the original transformer we had absolute positional embeddings.

What happened is we just add a learned position vector to the token embedding at the very beginning of the network before the input enters the first layer:

$$
x_{\text{input}} = x_{\text{word}} + p_{\text{pos}}
$$

The position vector is just a 1D vector of numbers with the exact same dimension as the token embedding (`hidden_dim`).

The model creates a giant 2D matrix called the positional embedding with shape:

`(max_seq_len, hidden_dim)`

If your input tokens are `["The", "cat", "sat"]`, their position indices are `[0, 1, 2]`.

You look up row 0, row 1, and row 2, and add them element-by-element to the token vectors.

The weights were initialized randomly and adjusted during training via gradient descent.

---

## Sinusoidal Absolute Embeddings

Also see sinusoidal absolute embeddings.

For a single position `pos`, the 576-dimensional vector alternates between sine and cosine values at geometrically decaying frequencies.

Instead of learning a table, the original paper filled the position vectors using predetermined math formulas made of sine and cosine waves.

The position vector is:

$$
\text{Pos vector} =
[
\sin(pos \cdot w_0),
\cos(pos \cdot w_0),
\sin(pos \cdot w_1),
\cos(pos \cdot w_1),
\dots,
\sin(pos \cdot w_k),
\cos(pos \cdot w_k)
]
$$

`k` is half of the hidden dim since we are using sin and cos pairs (2 values for a single frequency index).

The `w` values are the frequencies of the sine and cosine waves.

- **High frequency** → short wavelength and the model learns immediate local patterns (grammar).
- **Low frequency** → long wavelength and the model learns macro-level context (what paragraph are we in?).

---

# Why Absolute Embeddings Suck

- Relative distance is more important than the absolute index of a word in language.
- If a model is trained with `max_seq_len = 2048`, the model can't process token 2049; there's no vector for that position.
- If a big model with many layers that positional info is lost through non-linear activations, normalizations, residuals, etc.

Training on longer sequences isn't that simple because we are limited by compute and physical memory.

---

# Rotary Positional Embeddings (RoPE)

The key idea behind RoPE is instead of injecting positional information via adding a position vector with a token vector, we directly rotate the Q and K vectors in geometric space based on their position index before computing the dot product.

We do not rotate the V vectors because we don't want to distort the semantic content of our payload.

A token at position `m` is rotated by `m × θ` degrees.

A token at position `n` is rotated by `n × θ` degrees.

Looking at the formula for dot product, it's the product of the magnitude of the 2 vectors and the angle between them.

Before the rotation, the angle was `φ`.

After rotating Q by `m × θ` and K by `n × θ`, for example, the new angle is:

$$
(m - n)\theta + \phi
$$

The dot product now depends only on the relative distance between the 2 words.

Word A is at index 10 and word B at index 12 is the same rotation if A was at index 500 and B at 502 (`-2` for both).

---

## How RoPE Rotates the Head Dimensions

Important: Within the Query and Key activations we get from projecting our input, each attention head's representation for a given token gets rotated in its own 64-dimensional feature space before the dot products are computed.

In SmolLM, each head vector has `head_dim = 64`.

How do we rotate a 64-dimensional vector since rotation is fundamentally a 2D operation?

In math, every rotation in any number of dimensions is broken down into rotations within 2D planes.

So we treat the 64 numbers as **32 independent 2D pairs**.

Each pair rotates at a different speed/frequency just like clock hands. The rotation frequencies are computed using a formula.

- Pair 0 has a very large `theta` and rotates fast, letting the model distinguish adjacent words.
- Pair 31 rotates very slowly (`theta` is tiny) and it acts like the hour hand where it preserves long-range thematic associations across an entire document.

---

# Core Principles of RoPE

## 1. Geometric Rotation

Instead of adding an absolute embedding vector to the input tokens, RoPE rotates `Q` and `K` vectors in geometric space based on their sequence index `m`:

$$
Q_{\text{rot}} = \text{Rotate}(Q, m \cdot \theta)
$$

$$
K_{\text{rot}} = \text{Rotate}(K, n \cdot \theta)
$$

Because:

$$
u \cdot v = \Vert u \Vert \Vert v \Vert \cos(\phi)
$$

rotating Q by `mθ` and K by `nθ` makes their dot product depend strictly on the relative distance `(m - n)θ`:

$$
\text{Score}(m,n) \propto \cos\big((m-n)\theta\big)
$$

---

## 2. Applied Exclusively to Q and K

- **Q (queries)** and **K (keys)** determine *where* attention focuses.
- **V (values)** represents the *content* being retrieved and is **never** rotated.

---

## 3. Why 64D Is Rotated as 32 Independent 2D Pairs

- **Rotations are 2D operations:** A rotation occurs within a 2D plane (`x` and `y` swap components via `sin` and `cos` while preserving vector length).
- **Avoids full `64 × 64` dense matrix multiplications:** Block-diagonal structure allows element-wise execution:

$$
x_{\text{rot}} = x \odot \cos(\theta) + \tilde{x} \odot \sin(\theta)
$$

- **Preserves feature isolation:** Dimensions do not bleed across all 64 coordinates.

---

# The Frequency Spectrum: Fast vs. Slow Dimensions

The base frequency formula for pair index `i ∈ [0, 1, ..., 31]` is:

$$
\theta_i = \text{base}^{-2i/d},
\quad
\text{where } \text{base} = 10000,\ d = 64
$$

| Pair Index (`i`) | Frequency (`θᵢ`) | Rotation Speed | Behavioral Analogy | Function in Language |
| :--- | :--- | :--- | :--- | :--- |
| **`i = 0` (Early)** | ≈ 1.0 rad/token (57.3°) | **Very Fast** | Odometer *ones* digit / Clock *second hand* | **Local Syntax:** Sensitive to shifts of 1–2 tokens (e.g., word order, negation). |
| **`i = 16` (Middle)** | ≈ 0.01 rad/token | **Moderate** | Clock *minute hand* | **Phrase & Sentence Structure:** Balances relative distance over tens of tokens. |
| **`i = 31` (Late)** | ≈ 0.0001 rad/token | **Very Slow** | Odometer *ten-thousands* digit / Clock *hour hand* | **Global Context:** Barely rotates over 1,000+ tokens; maintains thematic signal. |

- **Short relative distances:** Fast dimensions oscillate sharply, enabling precise local syntax modeling.
- **Long relative distances:** Fast dimensions interfere destructively toward 0; slow-moving dimensions preserve consistent semantic links without attention exploding or vanishing.
