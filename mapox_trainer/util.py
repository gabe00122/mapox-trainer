from datetime import datetime
from typing import Any

import jax
from einops import rearrange
from flax import nnx
from mapox import TimeStep


def generate_unique_token() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


# Taken from pandas
def _normalise_json(
    data: Any,
    key_string: str,
    normalized_dict: dict[str, Any],
    separator: str,
) -> dict[str, Any]:
    """
    Main recursive function
    Designed for the most basic use case of pd.json_normalize(data)
    intended as a performance improvement, see #15621

    Parameters
    ----------
    data : Any
        Type dependent on types contained within nested Json
    key_string : str
        New key (with separator(s) in) for data
    normalized_dict : dict
        The new normalized/flattened Json dict
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar
    """
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{key_string}{separator}{key}"

            if not key_string:
                new_key = new_key.removeprefix(separator)

            _normalise_json(
                data=value,
                key_string=new_key,
                normalized_dict=normalized_dict,
                separator=separator,
            )
    else:
        normalized_dict[key_string] = data
    return normalized_dict


# Taken from pandas
def _normalise_json_ordered(data: dict[str, Any], separator: str) -> dict[str, Any]:
    """
    Order the top level keys and then recursively go to depth

    Parameters
    ----------
    data : dict or list of dicts
    separator : str, default '.'
        Nested records will generate names separated by sep,
        e.g., for sep='.', { 'foo' : { 'bar' : 0 } } -> foo.bar

    Returns
    -------
    dict or list of dicts, matching `normalised_json_object`
    """
    top_dict_ = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested_dict_ = _normalise_json(
        data={k: v for k, v in data.items() if isinstance(v, dict)},
        key_string="",
        normalized_dict={},
        separator=separator,
    )
    return {**top_dict_, **nested_dict_}


def json_normalize[T: (dict[str, Any] | list[dict[str, Any]])](
    ds: T,
    sep: str = "/",
) -> T:
    if isinstance(ds, tuple):
        ds = ds._asdict()

    normalised_json_object: dict[str, Any] = {}
    # expect a dictionary, as most jsons are. However, lists are perfectly valid
    if isinstance(ds, dict):
        normalised_json_object = _normalise_json_ordered(data=ds, separator=sep)
    elif isinstance(ds, list):
        normalised_json_list: list[dict[str, Any]] = [
            json_normalize(row, sep=sep) for row in ds
        ]  # type: ignore
        return normalised_json_list
    return normalised_json_object


def count_parameters(model: nnx.Module) -> str:
    params = nnx.state(model, nnx.Param)
    total_params = sum([x.size for x in jax.tree_util.tree_leaves(params)])
    return format_count(total_params)


def format_count(n: float) -> str:
    if not isinstance(n, (int, float)):
        raise TypeError("Input must be a number.")

    if n < 1000:
        # For numbers less than 1000, return as is.
        return str(n)
    elif n < 1_000_000:
        # Format for thousands (K).
        return f"{n / 1000:.2f}K"
    elif n < 1_000_000_000:
        # Format for millions (M).
        return f"{n / 1_000_000:.2f}M"
    else:
        # Format for billions (B).
        return f"{n / 1_000_000_000:.2f}B"


def lerp(a: jax.Array, b: jax.Array, progress: jax.Array) -> jax.Array:
    return a + progress * (b - a)


def add_seq_dim(ts: TimeStep):
    return jax.tree.map(lambda x: rearrange(x, "b ... -> b 1 ..."), ts)
