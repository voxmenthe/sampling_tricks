* Test out antislop sampler with Llama 3.2 3B
* Further customize the sampler in line with the jax implementation
* Do a research survey on sampling methods
* Bother _xjdr for some more ideas
* More experimentation with various sampling tricks and hyperparameters


So it should be kind of like Entropy of a conditional probability distribution of tokens using minimization-over-time sampling but actually aiming to maintain within constant entropy and varentropy bounds 

Should optionally use either a fixed entropy threshold to determine when to inject CoT tokens or a dynamic one that changes based on the appropriate entropy-related metrics.

In terms of "inject entropy based on branching" - it should bedoing this by, in some form, kind of changing the temperature and also, in some form, adding some noise distribution.

We also need to do a beam search over the top_k tokens at the branching point to increase the probability of a "good token" selection

The branching should be like returning the top_k tokens at that point and doing a beam search for as many tokens as it takes to try to get back within acceptable entropy bounds. Once its back, then you can resume the original entropy sampling style

When confident the outputs should generally not be affected.
