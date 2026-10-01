Normalization acts as a stabilizer for the model's data flow, forcing the data to fit into a standardized range. It prevents gradient vanishing or explosion as the data passes through many layers. 
This is important since vanishing or exploding gradients prevents the model from learning effectively.

Let's go a bit deeper into how gradients can explode or vanish. In an unnormalized transformer we could get a mix of both.
* vanishing gradients -> if gradients shrink as backprop moves through deeper layers, earlier layers will receive very small updates meaning they never improve
upon random weight initialization

- this happens locally at sepcific connections like softmax where a dominant logit pushes the function into its flat outer edges giving a very very small gradient

* exploding gradients -> here we have massive erratic adjustments to network weights and instead of converging to a minimal loss, it bounces around wildy or can jump to infinity (NaN)

- happens at linear/MLP steps since raw activation values inside the tokens have ballooned, and this acts as a massive multiplier on the backward pass




Before RMSNorm let's talk about BatchNorm and LayerNorm. 

BatchNorm -> 

LayerNorm ->

gamma is a learnable parameter 
