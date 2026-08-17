import jax
from einops import rearrange
from flax import nnx
from jax import numpy as jnp
from jax.typing import DTypeLike


class RnnBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.d_model = d_model
        self.rnn = nnx.RNN(
            nnx.GRUCell(
                d_model, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        )

    def initialize_carry(self, batch_size: int, rngs: nnx.Rngs):
        return self.rnn.cell.initialize_carry((batch_size, self.d_model), rngs)

    def __call__(self, inputs: jax.Array, seq_pos: jax.Array, carry=None):
        if carry is not None:
            x = rearrange(inputs, "b t x -> (b t) x")
            carry, x = self.rnn.cell(carry, x)
            x = x[:, None]
        else:
            x = self.rnn(inputs)

        return x, carry
