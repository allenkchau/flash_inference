In standard attention we project each word into a Q and K and compute the dot product. It's just multiplying numbers and adding them up.
A dot product measures semantic similarity not spatial order.

So this means for these sentences:
1. "Dog bites man"
2. "Man bites dog"

If there is no causal masking and no positional embeddings (like BERT-style attention) -> both sentences are just an unordered bag of words to the model, doesn't know who is biting who


Let's look at a longer sentence: "The dog that chased the three cats across the busy street barked."

If there is causal masking (GPT) but no positional embeddings -> words can only look backward but it still can't tell whether a past token was 1 step away or 49 steps away
- when computing attention for "barked" both "dog" and "cats" are valid nouns in the past
- without positional awareness, we can't distinguisht eh main subject
- also model can't natuarally learn syntatic rules like adjectives usually modifying the noun immediately adjacent to it

In the original transformer we had absolute positional embeddings.
What happened is we just add a learned position vector to the token embedding at the very beginning of the network before the input enters the first layer.
x_input = x_word + p_pos

The position vector is just a 1D vector of numbers with the exact same dimension as the token embedding (hidden_dim).

The model creates a giant 2D matrix called the weight positional embedding with shape (max_seq_len, hiddne_dim).
If your input tokens are ["The", "cat", "sat"], their position indices are [0, 1, 2].
You look up row 0, row 1, and row 2, and add them element-by-element to the token vectors.
The weights were initialized randomly and adjusted during training via gradient descent.



Also see sinusoidal absolute embeddings -> for a single position pos, the 576-dimensional vector alternates between sine and cosine values at geometrically decaying frequencies
Instead of learning a table, the original paper filled the position vectors using predetermined math formulas made of sine and cosine waves.

Pos vector = [ sin(pos * w_0), cos(pos * w_0), sin(pos * w_1), cos(pos * w_1), ..., sin(pos * w_k), cos(pos * w_k) ]; k is half of the hidden dim since we are using sin and cos pairs (2 values for a single freq index)
The w values are the frequencies of the sine and cosine waves. High frequency -> short wavelength and the model learns immediate local patterns (grammar)
Low frequency -> long wavelength and model learns macro-level context (what paragraph are we in?)



Why Absolute Embedddings suck:
- relative distance is more important than the absolute index of a word in language
- if a model is trained with max_seq_len=2048, the model can't process token 2049; there's no vector for that position
- if a big model with many layers that positional info is lost through non-linear ctivations, normalizations, residuals etc.

Training on longer sequences isn't that simple because we are limited by compute and physical memory.


The key idea behind RoPE is instead of injecting positional information via adding a position vector with a token vector, we directly rotate the Q and K vectors in geometric space based on their
position index before computing the dot product.

We do not rotate the V vectors because we don't want to distort the semantic content of our payload.


token at position m is rotated by m*theta degrees
token at position n is rotated by n*theta degrees

Looking at the formula for dot product, it's the product of the magnitude of the 2 vectors and the angle between them. Before the rotation the angle was phi.

After rotating Q by m*theta and K by n*theta for example, the new angle is (m - n)*theta + phi.

The dot product now depends only on the relative distance between the 2 words. Word A is at index 10 and word B at index 12 is the same rotation if A was at index 500 and B at 502 (-2 for both).

Important: Within the Query and Key activations we get from projecting our input, each attention head's representation for a given token gets rotated in its own 64 dimensional feature space before the dot products are computed.

In SmolLM, each head vector has head_dim=64. How do we rotate a 64 dimensional vector since rotation is fundamentally a 2D operation? In math, every rotation in any number of dimensions is broken down into rotations within 2D planes. 

So we treat the 64 numbers as 32 independent 2D pairs. Each pair rotates at a different speed/frequency just like clock hands. The rotation frequencies are computed using a formula.
