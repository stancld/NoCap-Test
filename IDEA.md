Adjustments to the original GPT-2 architecture in order to speed up training

1. **Replace GELU with SwiGLU** - While computationally SwiGLU is more expensive, it tends to be more
stable, provides better performance, and usually requiress less training steps.

2. **Grouped Query Attention with fused QKV projection**
