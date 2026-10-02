I will first explain the traditional multi-head attention and then expand a bit more on the grouped query attention variant that is used in SmolLM.

Say we have a sentence "The animal didn't cross the street because it was too tired".

At a high level self-attention allows the model to do things like associate "it" with "animal". In other words, we can look at other positions in the input sequence
for clues that can help lead to a better encoding for the word we are looking at.

Honestly to get a good understanding I would just read The Illustrated Transformer article by Jay Alammar.

query -> What am I looking for? This vector represents the current word that is actively searching the rest of the text for context.
key -> What do I offer? This vector acts like an indexing label or a tag for every word in the sequence, describing what kind of information that word contains.
value -> What content do I actually hold? Once the system matches the query with the keys, this vector contains the actual content or meaning that gets extracted and passed along.

GQ attention is a bit different. 

Motivation: During inference, the KV cache becomes very large. When generating a token at a time, the model stores the Keys and Values of previous tokens in memory.
So it can become very large.


