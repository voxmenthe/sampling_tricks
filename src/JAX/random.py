import jax
import jax.numpy as jnp

def prevent_doom_loop(logits: jax.Array, low_entropy_threshold: float = 1.0, noise_scale: float = 0.1, clip_logits: bool = True, clip_min: float = -10.0, clip_max: float = 10.0, key: jax.random.PRNGKey = jax.random.PRNGKey(1337)) -> jax.Array:

    probs = jax.nn.softmax(logits, axis=-1)

    # Calculate entropy of distribution
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)

    # Entropy too low == distribution too deterministic, add noise & apply to logits when entropy is low
    low_entropy_mask = entropy < low_entropy_threshold 
    noise = jax.random.normal(key, shape=logits.shape) * noise_scale
    modified_logits = jnp.where(low_entropy_mask[:, None], logits + noise, logits)
    
    if clip_logits:
        modified_logits = jnp.clip(modified_logits, a_min=clip_min, a_max=clip_max)

    return modified_logits

"""
    Features:
    - DRY Style Repetition Penalty
    - Dynamic Repetition Penalty
    - Contextual Repetition Penalty
    - Entropy based clipping
    - Entropy based Adaptive Temperature
    - Token Cluster boosting
    - Reasoning Chain Beam Search
    - Parallel CoT Sampling
"""