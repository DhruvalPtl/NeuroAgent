"""
src/models/registry.py
=======================
Global model registry — the single source of truth for which model
classes are available in the platform.

Design rationale
----------------
The registry decouples the orchestrator, code_writer, and leaderboard
from any specific model implementation.  Adding a new model (BiLSTM,
Transformer, ESM-2 fine-tune) in Milestone 2 requires:

  @register_model("bilstm")
  class BiLSTMModel(BaseModel):
      ...

That single decorator call is the ONLY change needed to make the new
model available everywhere in the platform — no wiring in orchestrator,
no configuration change, no import anywhere else.

Import-time registration
------------------------
Model modules must be *imported* for their @register_model decorators
to run.  The ``_ensure_models_registered()`` helper at the bottom of
this file handles this lazily.  Call it from any entry point that needs
a populated registry (launcher.py, tests, CLI tools).

Thread safety
-------------
The registry dict is populated at import time only and is read-only
thereafter.  No locking is required.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: Maps registry name (str) → model class (type[BaseModel]).
#: Populated at import time by @register_model decorators.
MODEL_REGISTRY: dict[str, type["BaseModel"]] = {}

# Names of all model modules under src/models/ that should be auto-imported.
# Update this list when adding a new model file.
_MODEL_MODULES = [
    "src.models.random_forest",
    "src.models.xgboost_model",
    "src.models.esm2_coral",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(name: str):
    """Class decorator that registers a BaseModel subclass.

    Parameters
    ----------
    name : str
        Registry key.  Must be unique across all registered models.
        Also used as the leaderboard display name and the key in
        tracking/db.py experiment records.

    Returns
    -------
    Callable
        The decorator function; returns the class unchanged so normal
        class definition syntax is preserved.

    Raises
    ------
    ValueError
        If ``name`` is already registered (prevents silent overwrite
        of an existing model by a code_writer attempt).

    Examples
    --------
    >>> @register_model("my_model")
    ... class MyModel(BaseModel):
    ...     name = "my_model"
    ...     ...
    """
    def decorator(cls: type) -> type:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"register_model: name {name!r} is already registered "
                f"(by {MODEL_REGISTRY[name].__qualname__}). "
                "Each model name must be globally unique."
            )
        if not hasattr(cls, "name"):
            raise TypeError(
                f"register_model: class {cls.__qualname__!r} must define a "
                "class-level 'name' attribute before registration."
            )
        if cls.name != name:
            raise ValueError(
                f"register_model: class {cls.__qualname__!r} has "
                f"name={cls.name!r} but is being registered under {name!r}. "
                "The class attribute 'name' must match the registry key."
            )
        MODEL_REGISTRY[name] = cls
        logger.debug("Registered model: %r → %s", name, cls.__qualname__)
        return cls
    return decorator


def get_model(name: str, **init_kwargs: Any) -> "BaseModel":
    """Instantiate and return a registered model by name.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"random_forest"``, ``"xgboost"``.
    **init_kwargs
        Constructor arguments forwarded to the model class.

    Returns
    -------
    BaseModel
        A fresh (untrained) model instance.

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.  The error message includes
        the full list of valid names so callers can self-correct.
    """
    _ensure_models_registered()

    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise KeyError(
            f"No model registered under name {name!r}.\n"
            f"Available models: {available}\n"
            "To add a new model, create a subclass of BaseModel, "
            "decorate it with @register_model('<name>'), and add its "
            "module to _MODEL_MODULES in src/models/registry.py."
        )
    cls = MODEL_REGISTRY[name]
    return cls(**init_kwargs)


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    _ensure_models_registered()
    return sorted(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_models_registered = False


def _ensure_models_registered() -> None:
    """Import all model modules so their @register_model decorators run.

    Idempotent — safe to call multiple times.
    """
    global _models_registered
    if _models_registered:
        return
    for module_path in _MODEL_MODULES:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(
                "Could not import model module %r: %s", module_path, exc
            )
    _models_registered = True
