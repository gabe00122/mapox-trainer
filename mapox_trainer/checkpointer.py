from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mapox_trainer.util import FixedParam


def _restore_args(state: nnx.State) -> ocp.args.StandardRestore:
    # everything is restored onto the first device, whichever it was saved from
    mesh = Mesh((jax.devices()[0],), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec())

    abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=sharding),
        state,
    )
    return ocp.args.StandardRestore(abstract_state)


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        # the model and the optimizer are separate items of a checkpoint. The
        # model keeps orbax's "default" name, which is where checkpoints saved
        # before the optimizer was added have it
        self.mngr = ocp.CheckpointManager(
            directory, item_names=("default", "optimizer")
        )
        self._filter = (nnx.Param, FixedParam)

    def save(self, model: object, optimizer: nnx.Optimizer, global_step: int):
        self.mngr.save(
            global_step,
            args=ocp.args.Composite(
                default=ocp.args.StandardSave(nnx.state(model, self._filter)),
                optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            ),
        )

    def restore[T](
        self, model: T, step: int, optimizer: nnx.Optimizer | None = None
    ) -> T:
        """Restore `model` in place, and `optimizer` too when one is given."""
        items = {"default": _restore_args(nnx.state(model, self._filter))}
        if optimizer is not None:
            items["optimizer"] = _restore_args(nnx.state(optimizer))

        restored = self.mngr.restore(step, args=ocp.args.Composite(**items))

        nnx.update(model, restored["default"])
        if optimizer is not None:
            nnx.update(optimizer, restored["optimizer"])
        return model

    def restore_latest[T](self, model: T, optimizer: nnx.Optimizer | None = None) -> T:
        return self.restore(model, self.mngr.latest_step() or 0, optimizer)

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
