import jax
import jax.numpy as jnp

def multinomial_sample_one(probs_sort: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    """
    Sample one token from a multinomial distribution from the given logits with sorted probabilities.

    Args:   
        probs_sort: A JAX array of shape (num_classes,) representing the sorted probabilities of the distribution.
        key: A JAX random key.

    Returns:
        A JAX array of shape (num_classes,) representing the sampled token.
    """
    q = jax.random_exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)


@jax.jit
def calculate_entropy(probs: jax.Array) -> jax.Array:
    """
    Calculate the entropy of a probability distribution.

    Args:
        probs: A JAX array of shape (batch_size, num_classes) representing the probabilities of the distribution.

    Returns:
        A JAX array of shape (batch_size,) representing the entropy of the distribution.
    """
    return -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), probs * jnp.log(probs), 0), axis=-1)


def prevent_doom_loop(
    logits: jax.Array, 
    low_entropy_threshold: float = 1.0, 
    noise_scale: float = 0.1,
    clip_logits: bool = True,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(1337)
) -> jax.Array:
    """
    Prevent the Doom Loop where the model keeps sampling the same token/set of tokens and won't produce an eot token.

    Args:
        logits: A JAX array of shape (batch_size, num_classes) representing the logits of the distribution.
        low_entropy_threshold: A float value representing the entropy threshold below which noise is added.
        noise_scale: A float value representing the scale of the noise to be added.
        key: A JAX random key.

    Returns:
        A JAX array of shape (batch_size, num_classes) representing the modified logits.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = calculate_entropy(probs)

    # Entropy too low == distribution too deterministic, add noise and apply to logits when entropy is low
    low_entropy_mask = entropy < low_entropy_threshold 
    noise = jax.random.normal(key, shape=logits.shape) * noise_scale
    modified_logits = jnp.where(low_entropy_mask[:, None], logits + noise, logits)
    
    # Optional: Clip the logits to prevent extreme values
    if clip_logits:
        modified_logits = jnp.clip(modified_logits, a_min=clip_min, a_max=clip_max)

    return modified_logits


def sample(logits: jax.Array, temperature: float = 0.666, top_p: float = 0.9, top_k: int = 27, key: jax.random.PRNGKey = jax.random.PRNGKey(1337)) -> jax.Array:
    """
    Sample tokens using temperature, top-k, and top-p sampling while preventing doom loops.

    Args:
        logits: A JAX array of shape (batch_size, num_classes) representing the logits of the distribution.
        temperature: A float value for temperature scaling.
        top_p: A float value for nucleus sampling threshold.
        top_k: An integer value for top-k sampling.
        key: A JAX random key.

    Returns:
        A JAX array of shape (batch_size, 1) representing the sampled tokens.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]

    # Prevent doom loop
    key, subkey = jax.random.split(key)
    logit = prevent_doom_loop(logit, subkey)

    # Apply temperature scaling
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    # Apply top-k sampling
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    probs_sort_jax = jnp.flip(top_k_probs, axis=-1)
    probs_idx_jax = jnp.flip(top_k_indices, axis=-1)
    probs_sum_jax = jnp.cumsum(probs_sort_jax, axis=-1)

    # Apply top-p sampling
    mask_jax = jnp.where(probs_sum_jax - probs_sort_jax > top_p, True, False)
    probs_sort_jax = probs_sort_jax * (1 - mask_jax)  # Set values to 0.0 using multiplication
    probs_sort_jax = probs_sort_jax / jnp.sum(probs_sort_jax, axis=-1, keepdims=True)  # Renormalize the probabilities

    # Sample one token from the sorted probabilities
    key, subkey = jax.random.split(key)
    next_token_jax = multinomial_sample_one(probs_sort_jax, subkey)
    next_token_g_jax = jnp.take_along_axis(probs_idx_jax, next_token_jax.reshape(bsz, 1), axis=-1)

    return next_token_g_jax.astype(jnp.int32)


def modern_sampler(
    logits: jax.Array,
    temperature: float = 0.666,
    top_p: float = 0.9,
    top_k: int = 27,
    key: jax.random.PRNGKey = jax.random.PRNGKey(1337),
    context_tokens: jax.Array = None,
    num_beams: int = 3,
    num_parallel_samples: int = 2,
    token_clusters: jax.Array = None,
    cluster_boost_factor: float = 1.2
) -> jax.Array:
    """
    Features:
    - DRY Style Repetition Penalty
    - Dynamic Repetition Penalty

    DRY Style and Dynamic Repetition Penalty: We use a dynamic penalty factor and apply it based on token frequency in the context, with a decay factor for older tokens.

    - Contextual Repetition Penalty

    Contextual Repetition Penalty: We apply an additional penalty to tokens that have appeared recently in the context.

    - Entropy based clipping
    - Entropy based Adaptive Temperature
    
    Entropy based clipping and Adaptive Temperature: We clip the logits to prevent extreme values and adjust the temperature based on the entropy of the distribution.
    
    - Token Cluster boosting

    Token Cluster Boosting: We boost the probabilities of tokens within the same semantic cluster as the highest probability token. This encourages the model to explore semantically similar alternatives, which can lead to more coherent and diverse outputs. The boosting is controlled by a cluster_boost_factor parameter.

    - Reasoning Chain Beam Search

    Reasoning Chain Beam Search: We implement a simple beam search to explore multiple possible next tokens and choose the best one based on cumulative probability.

    - Parallel CoT Sampling

    Parallel CoT Sampling: We generate multiple samples in parallel and choose the most diverse one to promote exploration.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]

    # DRY Style and Dynamic Repetition Penalty
    key, subkey = jax.random.split(key)
    penalty_factor = jax.random.uniform(subkey, minval=1.0, maxval=1.5)
    context_length = context_tokens.shape[1] if context_tokens is not None else 1
    penalty_decay = jnp.exp(-jnp.arange(context_length)[::-1] / 10)
    token_counts = jnp.sum(context_tokens[:, None] == jnp.arange(logit.shape[-1]), axis=1)
    penalty = 1 + (penalty_factor - 1) * token_counts * penalty_decay[:, None]
    logit = logit / penalty

    # Contextual Repetition Penalty
    if context_tokens is not None:
        recent_tokens = context_tokens[:, -5:]  # Consider last 5 tokens for context
        contextual_penalty = jnp.sum(recent_tokens[:, :, None] == jnp.arange(logit.shape[-1]), axis=1)
        logit = logit - contextual_penalty * 0.1  # Slight penalty for recent tokens

    # Entropy based clipping and Adaptive Temperature
    probs = jax.nn.softmax(logit, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1, keepdims=True)
    adaptive_temperature = jnp.where(entropy < 1.0, temperature * 0.9, temperature)
    logit = jnp.clip(logit, a_min=-10.0, a_max=10.0)  # Entropy based clipping
    logit = logit / adaptive_temperature

    # Token Cluster boosting

    
    # token_clusters: A JAX array that maps each token to its cluster ID.
    if token_clusters is not None:
        top_token = jnp.argmax(logit, axis=-1)
        top_token_cluster = jnp.take_along_axis(token_clusters, top_token[:, None], axis=-1)
        cluster_mask = (token_clusters == top_token_cluster)
        logit = jnp.where(cluster_mask, logit * cluster_boost_factor, logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = jax.lax.top_k(jax.nn.softmax(logit, axis=-1), k=top_k)
    probs_sort = jnp.flip(top_k_probs, axis=-1)
    probs_idx = jnp.flip(top_k_indices, axis=-1)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)

    # Apply top-p sampling
    mask = jnp.where(probs_sum - probs_sort > top_p, True, False)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)

    # Reasoning Chain Beam Search
    def beam_search_step(beam_logits, beam_key):
        beam_next_token = multinomial_sample_one(beam_logits, beam_key)
        return beam_next_token, beam_next_token

    key, *subkeys = jax.random.split(key, num=num_beams + 1)
    beam_tokens, _ = jax.lax.scan(beam_search_step, probs_sort, jnp.array(subkeys))
    beam_scores = jnp.sum(jnp.log(jnp.take_along_axis(probs_sort, beam_tokens, axis=-1)), axis=-1)
    best_beam = jnp.argmax(beam_scores, axis=-1)
    next_token = jnp.take_along_axis(beam_tokens, best_beam[:, None, None], axis=0).squeeze(1)

    # Parallel CoT Sampling
    def parallel_sample(parallel_key):
        return multinomial_sample_one(probs_sort, parallel_key)

    key, *parallel_keys = jax.random.split(key, num=num_parallel_samples + 1)
    parallel_tokens = jax.vmap(parallel_sample)(jnp.array(parallel_keys))
    
    # Choose the most diverse token from parallel samples
    token_diversity = jnp.sum(parallel_tokens[:, :, None] != parallel_tokens[:, None, :], axis=(1, 2))
    most_diverse_idx = jnp.argmax(token_diversity, axis=-1)
    next_token = jnp.take_along_axis(parallel_tokens, most_diverse_idx[:, None, None], axis=0).squeeze(1)

    return jnp.take_along_axis(probs_idx, next_token, axis=-1).astype(jnp.int32)