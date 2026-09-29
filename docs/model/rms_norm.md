RMSNorm acts as a stabilizer for the model's data flow. It prevents gradient vanishing or explosion as the data passes through many layers. This is important
since vanishing or exploding gradients prevents the model from learning effectively.

Little sidetrack:
* vanishing gradients -> if gradients shrink as backprop moves through deeper layers, earlier layers will receive very small updates meaning they never improve
upon random weight initialization

* exploding gradients -> here we have massive erratic adjustments to network weights and instead of converging to a minimal loss, it bounces around wildy or can jump to infinity (NaN)


Before RMSNorm let's talk about BatchNorm and LayerNorm. 

gamma is a learnable parameter 
