# Copyright 2025 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from simply.utils import position_encoding as pe_lib


class RoPETest(parameterized.TestCase):

  def test_preserves_shape(self):
    """RoPE.apply() should return same shape as input."""
    rope = pe_lib.RoPE()
    x = jnp.ones((2, 16, 4, 64))  # [batch, seq_len, num_heads, head_dim]
    result = rope.apply(x)
    self.assertEqual(result.shape, x.shape)

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_preserves_dtype(self, dtype):
    """RoPE should preserve dtype."""
    rope = pe_lib.RoPE()
    x = jnp.ones((2, 16, 4, 64), dtype=dtype)
    result = rope.apply(x)
    self.assertEqual(result.dtype, dtype)

  def test_position_dependent_output(self):
    """Same embedding at different positions should produce different outputs."""
    rope = pe_lib.RoPE()
    x = jnp.ones((1, 4, 1, 64))  # Same values at all positions
    result = rope.apply(x)
    # Position 0 and position 1 should differ
    self.assertFalse(jnp.allclose(result[0, 0], result[0, 1]))

  def test_custom_segment_positions(self):
    """Custom segment_positions should affect output."""
    rope = pe_lib.RoPE()
    x = jnp.ones((2, 4, 1, 64))

    # Sequential positions
    seq_pos = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    result_seq = rope.apply(x, segment_positions=seq_pos)

    # Reset positions (like packed sequences)
    reset_pos = jnp.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    result_reset = rope.apply(x, segment_positions=reset_pos)

    self.assertFalse(jnp.allclose(result_seq, result_reset))

  def test_different_max_timescales(self):
    """Different max_timescale should produce different outputs."""
    x = jnp.ones((1, 4, 1, 64))
    result_10k = pe_lib.RoPE(max_timescale=10_000).apply(x)
    result_1m = pe_lib.RoPE(max_timescale=1_000_000).apply(x)
    self.assertFalse(jnp.allclose(result_10k, result_1m))

  def test_different_scale_factors(self):
    """Different scale_factor should produce different outputs."""
    x = jnp.ones((1, 4, 1, 64))
    result_1x = pe_lib.RoPE(scale_factor=1.0).apply(x)
    result_8x = pe_lib.RoPE(scale_factor=8.0).apply(x)
    self.assertFalse(jnp.allclose(result_1x, result_8x))


if __name__ == '__main__':
  absltest.main()
