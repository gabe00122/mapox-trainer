from mapox import EnvironmentFactory

from mapox_trainer.envs.third_party.craftax_wrapper import (
    CraftaxConfig,
    CraftaxEnvironment,
)

__all__ = ["CraftaxConfig", "create_env_factory"]


def create_env_factory() -> EnvironmentFactory:
    factory = EnvironmentFactory()
    factory.register_env("craftax", CraftaxEnvironment)
    return factory
