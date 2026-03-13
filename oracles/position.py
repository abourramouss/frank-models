"""Position encoding oracles."""

import numpy as np


def rope(
    input: np.ndarray,
    positions: np.ndarray,
    freq_base: float = 10000.0,
    freq_scale: float = 1.0,
) -> np.ndarray:
    """Rotary Position Embeddings (RoPE).

    Uses the "rotate_half" convention (HuggingFace / LLaMA style):
    dimension i pairs with dimension i + head_dim/2.

    For position p and frequency freq[i]:
        out[..., i]             = input[..., i] * cos - input[..., i+half] * sin
        out[..., i + half_dim]  = input[..., i+half] * cos + input[..., i] * sin

    Args:
        input: Input tensor [batch, seq_len, n_head, head_dim]
        positions: Position indices [batch, seq_len]
        freq_base: Base for frequency computation (default 10000.0)
        freq_scale: Scale factor for frequencies (default 1.0)

    Returns:
        Rotated tensor [batch, seq_len, n_head, head_dim]
    """
    batch, seq_len, n_head, head_dim = input.shape
    half_dim = head_dim // 2

    # Compute frequencies: freq_scale / freq_base^(2i/head_dim)
    dim_indices = np.arange(half_dim)
    freqs = freq_scale / np.power(freq_base, 2 * dim_indices / head_dim)

    # Split into two halves (rotate_half convention)
    x_first = input[..., :half_dim]   # [batch, seq_len, n_head, half_dim]
    x_second = input[..., half_dim:]  # [batch, seq_len, n_head, half_dim]

    # Compute angles: positions[:, :, None, None] * freqs[None, None, None, :]
    angles = positions[:, :, None, None] * freqs[None, None, None, :]

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # Apply rotation (rotate_half style):
    # first_half'  = first_half * cos - second_half * sin
    # second_half' = second_half * cos + first_half * sin
    first_rot = x_first * cos_vals - x_second * sin_vals
    second_rot = x_second * cos_vals + x_first * sin_vals

    # Concatenate halves back to [batch, seq_len, n_head, head_dim]
    return np.concatenate([first_rot, second_rot], axis=-1)
