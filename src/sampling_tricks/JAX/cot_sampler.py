import jax
import jax.numpy as jnp
from typing import List, Tuple

def cot_sampler(
    logits: jax.Array,
    key: jax.random.PRNGKey,
    num_samples: int = 1,
    temperature: float = 1.0,
    entropy_threshold: float = 2.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3
) -> Tuple[jax.Array, jax.Array]:
    """
    Sampler which uses Entropy-based injection of CoT (Chain of Thought) tokens to tell the model to
    re-evaluate and inject entropy based on branching to arrive at the correct answer.

    Args:
        logits: A JAX array of shape (batch_size, vocab_size) representing the logits of the distribution.
        key: A JAX random key.
        num_samples: Number of samples to generate.
        temperature: Temperature for sampling.
        entropy_threshold: Threshold below which CoT tokens are injected.
        cot_tokens: List of token IDs representing CoT prompts (e.g., "Let's think step by step", "Let's check our work").
        max_cot_injections: Maximum number of CoT token injections allowed.

    Returns:
        A tuple containing:
        - A JAX array of shape (batch_size, num_samples) representing the sampled tokens.
        - A JAX array of shape (batch_size, num_samples) indicating whether a CoT token was injected (1) or not (0).
    """
    batch_size, vocab_size = logits.shape

    if cot_tokens is None:
        cot_tokens = [100, 101, 102]  # Example token IDs for CoT prompts

    def sample_with_cot(carry, _):
        current_logits, current_key, cot_injection_count = carry
        
        # Calculate entropy
        probs = jax.nn.softmax(current_logits / temperature, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        
        # Decide whether to inject CoT token
        inject_cot = (entropy < entropy_threshold) & (cot_injection_count < max_cot_injections)
        
        # Sample token
        key1, key2 = jax.random.split(current_key)
        sampled_token = jax.random.categorical(key1, current_logits / temperature, axis=-1)
        
        # If injecting CoT, replace sampled token with a random CoT token
        cot_token = jax.random.choice(key2, jnp.array(cot_tokens))
        final_token = jnp.where(inject_cot, cot_token, sampled_token)
        
        # Update CoT injection count
        new_cot_injection_count = cot_injection_count + inject_cot.astype(jnp.int32)
        
        return (current_logits, key2, new_cot_injection_count), (final_token, inject_cot.astype(jnp.int32))

    keys = jax.random.split(key, num_samples)
    initial_carry = (logits, keys[0], jnp.zeros(batch_size, dtype=jnp.int32))
    _, (sampled_tokens, cot_injected) = jax.lax.scan(sample_with_cot, initial_carry, None, length=num_samples)

    return sampled_tokens.T, cot_injected.T

@jax.jit
def entropy_based_cot_sampling(
    logits: jax.Array,
    key: jax.random.PRNGKey,
    num_samples: int = 1,
    temperature: float = 1.0,
    entropy_threshold: float = 2.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-compiled version of the cot_sampler function for faster execution.
    """
    return cot_sampler(logits, key, num_samples, temperature, entropy_threshold, cot_tokens, max_cot_injections)

def generate_with_cot(
    model_fn,
    initial_tokens: jax.Array,
    max_length: int,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
    entropy_threshold: float = 2.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3
) -> jax.Array:
    """
    Generate a sequence using the model function and CoT sampling.

    Args:
        model_fn: A function that takes tokens and returns logits.
        initial_tokens: Initial token sequence.
        max_length: Maximum length of the generated sequence.
        key: A JAX random key.
        temperature: Temperature for sampling.
        entropy_threshold: Threshold for CoT injection.
        cot_tokens: List of CoT token IDs.
        max_cot_injections: Maximum number of CoT injections.

    Returns:
        A JAX array containing the generated sequence.
    """
    def body_fn(carry, _):
        tokens, current_key, cot_injection_count = carry
        logits = model_fn(tokens)
        key1, key2 = jax.random.split(current_key)
        new_token, cot_injected = entropy_based_cot_sampling(
            logits[:, -1:],
            key1,
            num_samples=1,
            temperature=temperature,
            entropy_threshold=entropy_threshold,
            cot_tokens=cot_tokens,
            max_cot_injections=max_cot_injections - cot_injection_count
        )
        new_tokens = jnp.concatenate([tokens, new_token.T], axis=1)
        new_cot_injection_count = cot_injection_count + cot_injected[0, 0]
        return (new_tokens, key2, new_cot_injection_count), new_tokens

    initial_carry = (initial_tokens, key, jnp.array(0, dtype=jnp.int32))
    _, generated_tokens = jax.lax.scan(
        body_fn,
        initial_carry,
        None,
        length=max_length - initial_tokens.shape[1]
    )
    
    return generated_tokens[-1]
