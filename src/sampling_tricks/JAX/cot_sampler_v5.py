import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable, Dict
from functools import partial

@jax.jit
def calculate_entropy(probs: jax.Array) -> jax.Array:
    return -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)

def calculate_varentropy(entropy_history: jax.Array) -> float:
    return jnp.var(entropy_history)

def adaptive_temperature(entropy: float, target_entropy: float, min_temp: float = 0.1, max_temp: float = 2.0) -> float:
    return jnp.clip(target_entropy / (entropy + 1e-5), min_temp, max_temp)

def beam_search(logits: jax.Array, beam_width: int, max_steps: int, model_fn: Callable, temperature: float) -> jax.Array:
    def beam_step(beam_state, _):
        current_sequences, current_scores, current_logits = beam_state
        next_token_logits = jax.vmap(model_fn)(current_sequences)
        next_token_probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
        next_token_log_probs = jnp.log(next_token_probs)
        candidate_scores = current_scores[:, None] + next_token_log_probs
        top_k_scores, top_k_indices = jax.lax.top_k(candidate_scores.reshape(-1), k=beam_width)
        best_sequences = current_sequences[top_k_indices // next_token_probs.shape[-1]]
        best_tokens = top_k_indices % next_token_probs.shape[-1]
        new_sequences = jnp.concatenate([best_sequences, best_tokens[:, None]], axis=1)
        return (new_sequences, top_k_scores, next_token_logits), None

    initial_sequences = jnp.argmax(logits, axis=-1)[:beam_width, None]
    initial_scores = jnp.zeros(beam_width)
    initial_state = (initial_sequences, initial_scores, logits)
    final_state, _ = jax.lax.scan(beam_step, initial_state, None, length=max_steps)
    best_sequence = final_state[0][0]
    return best_sequence

def dynamic_entropy_threshold(current_entropy: float, history: jax.Array, alpha: float = 0.1) -> float:
    moving_avg = jnp.mean(history)
    return alpha * current_entropy + (1 - alpha) * moving_avg

def inject_noise(logits: jax.Array, noise_scale: float, key: jax.random.PRNGKey) -> jax.Array:
    noise = jax.random.normal(key, shape=logits.shape) * noise_scale
    return logits + noise

def cot_sampler_v5(
    logits: jax.Array,
    key: jax.random.PRNGKey,
    model_fn: Callable,
    num_samples: int = 1,
    temperature: float = 1.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3,
    injection_cooldown: int = 10,
    punctuation_tokens: List[int] = None,
    beam_width: int = 5,
    max_beam_steps: int = 10,
    entropy_threshold: float = 2.0,
    varentropy_threshold: float = 0.5,
    backspace_penalty: float = 0.1,
    use_dynamic_threshold: bool = True,
    noise_scale: float = 0.1,
    target_entropy: float = 2.0,
    entropy_history_size: int = 20,
    quadrant_weights: Dict[str, float] = None,
    adaptive_temp_range: Tuple[float, float] = (0.1, 2.0)
) -> Tuple[jax.Array, jax.Array]:
    """
    Advanced CoT (Chain of Thought) sampler with improved steerability and quality of CoT outputs.

    This sampler implements a sophisticated approach to token generation by considering
    both the entropy of the current token distribution and the variance of entropy over time
    (varentropy). It dynamically switches between four strategies based on the current state:

    1. Argmax (Low Entropy, Low Varentropy): Use argmax sampling for high confidence.
    2. Branch (High Entropy, High Varentropy): Use beam search with adaptive temperature and noise injection.
    3. CoT (Chain of Thought) (High Entropy, Low Varentropy): Inject CoT tokens to encourage step-by-step reasoning.
    4. Backspace (Low Entropy, High Varentropy): Implement a backtracking mechanism to explore alternative paths.

    Args:
        logits: A JAX array of shape (batch_size, vocab_size) representing the logits of the distribution.
        key: A JAX random key.
        model_fn: A function that takes tokens and returns logits for the next token.
        num_samples: Number of samples to generate.
        temperature: Base temperature for sampling.
        cot_tokens: List of token IDs representing CoT prompts.
        max_cot_injections: Maximum number of CoT token injections allowed.
        injection_cooldown: Minimum number of tokens between CoT injections.
        punctuation_tokens: List of token IDs representing punctuation marks.
        beam_width: Width of the beam for beam search during branching.
        max_beam_steps: Maximum number of steps for beam search.
        entropy_threshold: Base threshold to distinguish between high and low entropy.
        varentropy_threshold: Threshold to distinguish between high and low varentropy.
        backspace_penalty: Penalty factor for the backspace action.
        use_dynamic_threshold: Whether to use a dynamic entropy threshold.
        noise_scale: Scale of the noise to add during branching.
        target_entropy: Target entropy for adaptive temperature scaling.
        entropy_history_size: Size of the entropy history buffer for calculating varentropy.
        quadrant_weights: Dictionary of weights for each quadrant strategy.
        adaptive_temp_range: Tuple of (min_temp, max_temp) for adaptive temperature scaling.

    Returns:
        A tuple containing:
        - A JAX array of shape (batch_size, num_samples) representing the sampled tokens.
        - A JAX array of shape (batch_size, num_samples) indicating the sampling strategy used.
    """
    batch_size, vocab_size = logits.shape

    if cot_tokens is None:
        cot_tokens = [100, 101, 102]  # Example token IDs for CoT prompts
    
    if punctuation_tokens is None:
        punctuation_tokens = [1, 2, 3]  # Example token IDs for punctuation marks

    if quadrant_weights is None:
        quadrant_weights = {"argmax": 1.0, "branch": 1.0, "cot": 1.0, "backspace": 1.0}

    def is_appropriate_injection_point(previous_token, tokens_since_last_injection):
        return (previous_token in punctuation_tokens) and (tokens_since_last_injection >= injection_cooldown)

    def sample_with_quadrants(carry, _):
        current_logits, current_key, cot_injection_count, previous_token, tokens_since_last_injection, entropy_history = carry
        
        # Calculate entropy and probabilities
        probs = jax.nn.softmax(current_logits / temperature, axis=-1)
        entropy = calculate_entropy(probs)
        varentropy = calculate_varentropy(entropy_history)
        
        # Determine the current quadrant
        current_threshold = dynamic_entropy_threshold(entropy, entropy_history) if use_dynamic_threshold else entropy_threshold
        low_entropy = entropy < current_threshold
        low_varentropy = varentropy < varentropy_threshold
        
        # Adaptive temperature
        adaptive_temp = adaptive_temperature(entropy, target_entropy, *adaptive_temp_range)
        
        # Sampling strategies for each quadrant
        def argmax_strategy():
            return jnp.argmax(current_logits, axis=-1)
        
        def branch_strategy():
            key1, key2 = jax.random.split(current_key)
            noisy_logits = inject_noise(current_logits, noise_scale, key1)
            return beam_search(noisy_logits, beam_width, max_beam_steps, model_fn, adaptive_temp)[-1]
        
        def cot_strategy():
            appropriate_point = is_appropriate_injection_point(previous_token, tokens_since_last_injection)
            inject_cot = (cot_injection_count < max_cot_injections) & appropriate_point
            cot_token = jax.random.choice(current_key, jnp.array(cot_tokens))
            return jnp.where(inject_cot, cot_token, jax.random.categorical(current_key, current_logits / adaptive_temp, axis=-1))
        
        def backspace_strategy():
            backspaced_logits = current_logits.at[previous_token].add(-backspace_penalty)
            return jax.random.categorical(current_key, backspaced_logits / adaptive_temp, axis=-1)
        
        # Choose the appropriate strategy with weighted probabilities
        strategy_probs = jnp.array([
            quadrant_weights["argmax"] if low_entropy and low_varentropy else 0,
            quadrant_weights["branch"] if not low_entropy and not low_varentropy else 0,
            quadrant_weights["cot"] if not low_entropy and low_varentropy else 0,
            quadrant_weights["backspace"] if low_entropy and not low_varentropy else 0
        ])
        strategy_probs /= jnp.sum(strategy_probs)
        
        strategy_key, token_key = jax.random.split(current_key)
        chosen_strategy = jax.random.choice(strategy_key, 4, p=strategy_probs)
        
        final_token = jax.lax.switch(chosen_strategy, [argmax_strategy, branch_strategy, cot_strategy, backspace_strategy])
        
        # Update CoT injection count and tokens since last injection
        new_cot_injection_count = cot_injection_count + (chosen_strategy == 2).astype(jnp.int32)
        new_tokens_since_last_injection = jnp.where(chosen_strategy == 2, 0, tokens_since_last_injection + 1)
        
        # Update entropy history
        new_entropy_history = jnp.concatenate([entropy_history[1:], jnp.array([entropy])])
        
        new_carry = (current_logits, token_key, new_cot_injection_count, final_token, new_tokens_since_last_injection, new_entropy_history)
        return new_carry, (final_token, chosen_strategy)

    keys = jax.random.split(key, num_samples)
    initial_carry = (
        logits,
        keys[0],
        jnp.zeros(batch_size, dtype=jnp.int32),
        jnp.full(batch_size, -1, dtype=jnp.int32),
        jnp.full(batch_size, injection_cooldown, dtype=jnp.int32),
        jnp.full((entropy_history_size,), entropy_threshold)
    )
    _, (sampled_tokens, strategies_used) = jax.lax.scan(sample_with_quadrants, initial_carry, None, length=num_samples)

    return sampled_tokens.T, strategies_used.T

@partial(jax.jit, static_argnums=(2,))
def entropy_based_quadrant_sampling_v5(
    logits: jax.Array,
    key: jax.random.PRNGKey,
    model_fn: Callable,
    num_samples: int = 1,
    temperature: float = 1.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3,
    injection_cooldown: int = 10,
    punctuation_tokens: List[int] = None,
    beam_width: int = 5,
    max_beam_steps: int = 10,
    entropy_threshold: float = 2.0,
    varentropy_threshold: float = 0.5,
    backspace_penalty: float = 0.1,
    use_dynamic_threshold: bool = True,
    noise_scale: float = 0.1,
    target_entropy: float = 2.0,
    entropy_history_size: int = 20,
    quadrant_weights: Dict[str, float] = None,
    adaptive_temp_range: Tuple[float, float] = (0.1, 2.0)
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-compiled version of the advanced cot_sampler_v5 function for faster execution.
    """
    return cot_sampler_v5(
        logits, key, model_fn, num_samples, temperature, cot_tokens,
        max_cot_injections, injection_cooldown, punctuation_tokens,
        beam_width, max_beam_steps, entropy_threshold, varentropy_threshold, backspace_penalty,
        use_dynamic_threshold, noise_scale, target_entropy, entropy_history_size,
        quadrant_weights, adaptive_temp_range
    )

def generate_with_quadrant_cot_v5(
    model_fn: Callable,
    initial_tokens: jax.Array,
    max_length: int,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
    cot_tokens: List[int] = None,
    max_cot_injections: int = 3,
    injection_cooldown: int = 10,
    punctuation_tokens: List[int] = None,
    beam_width: int = 5,
    max_beam_steps: int = 10,
    entropy_threshold: float = 2.0,
    varentropy_threshold: float = 0.5,
    backspace_penalty: float = 0.1,
    use_dynamic_threshold: bool = True,
    noise_scale: float = 0.1,
    target_entropy: float = 2.0,
    entropy_history_size: int = 20,
    quadrant_weights: Dict[str, float] = None,
    adaptive_temp_range: Tuple[float, float] = (0.1, 2.0)
) -> Tuple[jax.Array, jax.Array]:
    """
    Generate a sequence using the model function and advanced quadrant-based CoT sampling.

    This function uses the advanced quadrant-based CoT sampler to generate a sequence of tokens
    while dynamically adapting the sampling strategy based on entropy and varentropy.

    Args:
        (... same as cot_sampler_v5 function ...)

    Returns:
        A tuple containing:
        - A JAX array containing the generated sequence.
        - A JAX array containing the strategies used for each token.
    """
    def body_fn(carry, _):
        tokens, current_key, cot_injection_count, previous_token, tokens_since_last_injection, entropy_history = carry
        logits = model_fn(tokens)
        key1, key2 = jax.random.split(current_key)
        new_token, strategy_used = entropy_based_quadrant_sampling_v5(
            logits[:, -1],
            key1,
            model_fn,
            num_samples=1,
            temperature=temperature,
            cot_tokens=cot_tokens,
            max_cot_injections=max_cot_injections - cot_injection_count,
            injection_cooldown=injection_cooldown,
            punctuation_tokens=punctuation_tokens,
            beam_width=beam_width,
            max_beam_steps=max_beam_steps,
            entropy_threshold=entropy_threshold,
            varentropy_threshold=varentropy_threshold,
            backspace_penalty=backspace_penalty,
            use_dynamic_threshold=use_dynamic_threshold,
            noise_scale=noise_scale,
            target_entropy=target_entropy,
            entropy_history_size=entropy_history_size,
            quadrant_weights=quadrant_weights,
            adaptive_temp_range=adaptive_temp_range
        )
        new_tokens = jnp.concatenate([tokens, new_token], axis=-1)
        new_cot_injection_count = cot_injection_count + (strategy_used == 2).astype(jnp.int32)
        new_tokens_since_last_injection = jnp.where(strategy_used == 2, 0, tokens_since_last_injection + 1)
        new_entropy_history = jnp.concatenate([entropy_history[1:], jnp.array([calculate_entropy(jax.nn.softmax(logits[:, -1] / temperature, axis=-1))])])
        return (new_tokens, key2, new_cot_injection_count, new_token, new_tokens_since_last_injection, new_entropy_history), (new_token, strategy_used)

    initial_carry = (
        initial_tokens,
        key,
        jnp.zeros(initial_tokens.shape[0], dtype=jnp.int32),
        jnp.full(initial_tokens.shape[0], -1, dtype=jnp.int32),
        jnp.full(initial_tokens.shape[0], injection_cooldown, dtype=jnp.int32),
        jnp.full((initial_tokens.shape[0], entropy_history_size), entropy_threshold)
    )
    _, (generated_tokens, strategies_used) = jax.lax.scan(body_fn, initial_carry, None, length=max_length - initial_tokens.shape[1])

    return jnp.concatenate([initial_tokens, generated_tokens.squeeze(axis=-1)], axis=-1), strategies_used.squeeze(axis=-1)