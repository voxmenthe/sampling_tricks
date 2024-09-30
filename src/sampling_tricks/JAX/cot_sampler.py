import jax
import jax.numpy as jnp

def cot_sampler(logits: jax.Array, key: jax.random.PRNGKey, num_samples: int = 1) -> jax.Array:
    """
    Sampler which uses Entropy-based injection of CoT (Chain of Thought) tokens to tell the model to
    re-evaluate and inject entropy based on branching to arrive at the correct answer.
    """
    