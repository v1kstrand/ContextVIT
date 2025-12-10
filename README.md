# ContextVIT – Sub-Quadratic Attention

This repo explores sub-quadratic ways to perform the attention operation in a Vision Transformer (ViT — Vision Transformer) by reducing the effective number of key–value (KV) tokens each query needs to attend to.

We are currently experimenting with two main ideas:

1. **Tiled token grid**
   - Tokens are “tiled” into a fixed grid.
   - Attention is computed on this tiled representation by letting each query attend to the tiles instead of all individual tokens.
   - Intuition: summarize local neighborhoods into a smaller set of tiles while preserving enough structure for downstream layers.

2. **Learnable KV clustering**
   - A learnable clustering function groups KV tokens into a smaller set of clusters.
   - Each query attends to these clusters rather than every token.
   - Intuition: let the model learn meaningful context groups and use them as a compressed representation of the sequence.

This codebase is intended as a research sandbox rather than a polished library.  
Results, experiments, and training setups will be documented and cleaned up later.

If you are interested in the details or would like to discuss the ideas, please feel free to reach out by email.
