# flake8: noqa

temperature = 1.0
def dynamic_entropy_threshold(entropy, entropy_history):
    return jnp.mean(entropy_history)
def calculate_entropy(probs):
    return -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), probs * jnp.log(probs), 0), axis=-1)
use_dynamic_threshold = False
noise_scale = 0.1
branch_threshold = 0.1
entropy_bounds = (0.1, 0.5)
injection_cooldown = 10
max_cot_injections = 10
punctuation_tokens = ['.', '!', '?']
beam_width = 5
max_beam_steps = 10
model_fn = None
entropy_threshold = 0.1
import jax
import jax.numpy as jnp
def beam_search(logits, beam_width, max_beam_steps, model_fn, temperature):
    batch_size = logits.shape[0]
    beam = jnp.full((batch_size, beam_width), -1, dtype=jnp.int32)
    beam_scores = jnp.zeros((batch_size, beam_width))
    beam_scores = jnp.where(beam == -1, -jnp.inf, beam_scores)
    beam_scores = beam_scores + jnp.log(temperature)
    beam_scores = jnp.where(beam == -1, -jnp.inf, beam_scores)
    beam_scores = beam_scores + jnp.log(temperature)
    return beam, beam_scores

def is_appropriate_injection_point(previous_token, tokens_since_last_injection):
    return (previous_token in punctuation_tokens) and (tokens_since_last_injection >= injection_cooldown)

def should_branch(probs, entropy):
    top_two_probs = jnp.sort(probs)[-2:]
    return ((top_two_probs[1] - top_two_probs[0]) < branch_threshold) | (entropy < entropy_bounds[0]) | (entropy > entropy_bounds[1])

def sample_with_cot(carry, _):
    current_logits, current_key, cot_injection_count, previous_token, tokens_since_last_injection, entropy_history = carry

    probs = jax.nn.softmax(current_logits / temperature, axis=-1)
    entropy = calculate_entropy(probs)
    
    current_threshold = dynamic_entropy_threshold(entropy, entropy_history) if use_dynamic_threshold else entropy_threshold
    
    # Decide whether to inject CoT token
    appropriate_point = is_appropriate_injection_point(previous_token, tokens_since_last_injection)
    inject_cot = (entropy < current_threshold) & (cot_injection_count < max_cot_injections) & appropriate_point
    
    # Branching logic
    branch = should_branch(probs, entropy)
    if branch:
        # Adjust temperature and add noise
        branching_temperature = temperature * (1 + jax.random.uniform(current_key, minval=-0.1, maxval=0.1))
        noise = jax.random.normal(current_key, shape=current_logits.shape) * noise_scale
        noisy_logits = current_logits + noise
        
        # Perform beam search
        best_sequence = beam_search(noisy_logits, beam_width, max_beam_steps, model_fn, branching_temperature)
        final_token = best_sequence[-1]
    else:
        # Regular sampling
        final_token = jax.random.categorical(current_key, current_logits / temperature, axis=-1)
    
    # If injecting CoT, replace sampled token with a random CoT token
    cot_token = jax.random.choice(current_key, jnp.array(cot_tokens))
    final_token = jnp.where(inject_cot, cot_token, final_token)
    
    # Update CoT injection count and tokens since last injection
    new_cot_injection_count = cot_injection_count + inject_cot.astype(jnp.int32)
    new_tokens_since_last_injection = jnp.where(inject_cot, 0, tokens_since_last_injection + 1)
    
    # Update entropy history
    new_entropy_history = jnp.concatenate([entropy_history[1:], jnp.array([entropy])])
    
    new_carry = (current_logits, current_key, new_cot_injection_count, final_token, new_tokens_since_last_injection, new_entropy_history)
    return new_carry, (final_token, inject_cot.astype(jnp.int32))


################################################################################
def dynamic_entropy_threshold(current_entropy: float, history: List[float], alpha: float = 0.1) -> float:
    """
    Calculate a dynamic entropy threshold based on the history of entropies.

    Args:
        current_entropy: The current entropy value.
        history: A list of previous entropy values.
        alpha: The smoothing factor for the moving average.

    Returns:
        A float representing the dynamic entropy threshold.
    """
    if not history:
        return current_entropy
    moving_avg = sum(history) / len(history)
    return alpha * current_entropy + (1 - alpha) * moving_avg

################################################################################
def should_branch(probs, entropy):
    top_two_probs = jnp.sort(probs)[-2:]
    return ((top_two_probs[1] - top_two_probs[0]) < branch_threshold) | (entropy < entropy_bounds[0]) | (entropy > entropy_bounds[1])

################################################################################

        # Update dynamic threshold if enabled
        current_threshold = dynamic_entropy_threshold(entropy, entropy_history) if use_dynamic_threshold else entropy_threshold