import flax
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh
from mapox_trainer.util import FixedParam


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model: object, global_step: int):
        state = nnx.state(model, nnx.Param)
        self.mngr.save(global_step, args=ocp.args.StandardSave(state))

    def restore[T](self, model: T, step: int) -> T:
        device = jax.devices()[0]
        mesh = Mesh((device,), ("batch",))

        value_state = nnx.state(model, (nnx.Param, FixedParam))
        abstract_state = jax.tree.map(
            lambda x, s: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=s),
            value_state,
            nnx.get_named_sharding(value_state, mesh),
        )
        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )

        nnx.update(model, restored_state)
        return model

    def restore_latest[T](self, model: T) -> T:
        return self.restore(model, self.mngr.latest_step() or 0)

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
