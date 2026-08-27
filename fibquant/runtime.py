"""Explicit, reversible install lifecycle for the FibQuant runtime patches.

`enable_fibquant`'s predecessor was a one-shot, process-global, irreversible
patch: a module-level boolean guard made the second call — with a *different*
spec — return silently (the notebook re-run footgun), the `model` argument
was ignored (the class was always patched), `model=None` patched every model
instantiated afterwards, nothing could be uninstalled, and generate() built a
plain DynamicCache so compression silently skipped the generation path.

`FibQuantRuntime` replaces it with an explicit lifecycle:

  - `install(model=None)`: class-level install (patches the target model
    class, covering forward() and the generate cache factory); with a model,
    that instance only.
  - `install()` with the same spec is idempotent; with a *different* spec it
    re-patches and logs a warning — never silent.
  - `uninstall()` restores the original methods (test isolation, teardown).
  - `active_spec` / `active_specs()` make the current state inspectable.

Both patch points are installed and removed together, so there is no
half-enabled state where forward() is compressed but generate() is not:

  - the model class's `forward()`: builds a FibQuantCache when use_cache is
    set and no cache was passed (plain forward calls, lm-eval loglikelihood);
  - the model class's `_prepare_cache_for_generation()` (the cache factory
    generate() calls in transformers 5.x): replaces the plain DynamicCache it
    just built with a FibQuantCache. User-supplied caches, static/quantized/
    offloaded implementations, and assisted generation are left untouched.

The target class is injectable so tests exercise the exact patch contract on
a ~20-line dummy class instead of loading Qwen3.5.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Any

import torch

from .spec import FibQuantSpec

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_CLASS: type | None = None  # resolved lazily on first class-level install


@dataclass
class _Patch:
    spec: FibQuantSpec
    target: Any  # the model class (class-level) or the model instance
    is_class_level: bool
    original_forward: Any
    original_prepare: Any


# One active patch per target; keyed by id(target) so instances and classes
# never collide. install()/uninstall() are the only state transitions.
_INSTALLED: dict[int, _Patch] = {}


def _default_model_class() -> type:
    global _DEFAULT_MODEL_CLASS
    if _DEFAULT_MODEL_CLASS is None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        _DEFAULT_MODEL_CLASS = Qwen3_5TextModel
    return _DEFAULT_MODEL_CLASS


def _new_cache(config: Any, spec: FibQuantSpec) -> Any:
    from .cache import FibQuantCache

    return FibQuantCache(config=config, spec=spec)


def _make_wrappers(original_forward, original_prepare, spec: FibQuantSpec):
    """Build the wrapped forward + generate-cache factory around the originals."""

    def wrapped_forward(self, *args, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        use_cache = kwargs.get("use_cache")
        if use_cache and past_key_values is None:
            kwargs["past_key_values"] = _new_cache(self.config, spec)
        return original_forward(self, *args, **kwargs)

    def wrapped_prepare(self, generation_config, model_kwargs, *args, **kwargs):
        result = original_prepare(self, generation_config, model_kwargs, *args, **kwargs)
        # generate() built a plain DynamicCache (cache_implementation None or
        # "dynamic"). Replace it with a FibQuantCache so compression actually
        # covers generation; anything else (user-supplied cache, static,
        # quantized, offloaded, assisted-generation recording) is untouched.
        if not getattr(generation_config, "is_assistant", False):
            from transformers.cache_utils import DynamicCache

            cache = model_kwargs.get("past_key_values")
            if type(cache) is DynamicCache:
                model_kwargs["past_key_values"] = _new_cache(self.config, spec)
        return result

    return wrapped_forward, wrapped_prepare


def _install(target: Any, spec: FibQuantSpec, is_class_level: bool) -> _Patch:
    # Capture UNBOUND functions: target may be an instance, and attribute
    # lookup on an instance binds methods (double-binding would shift every
    # argument of the wrapper's call below).
    cls = target if isinstance(target, type) else type(target)
    original_forward = cls.forward
    original_prepare = cls._prepare_cache_for_generation
    wrapped_forward, wrapped_prepare = _make_wrappers(original_forward, original_prepare, spec)
    if is_class_level:
        target.forward = wrapped_forward
        target._prepare_cache_for_generation = wrapped_prepare
    else:  # instance attributes shadow the class methods
        target.forward = types.MethodType(wrapped_forward, target)
        target._prepare_cache_for_generation = types.MethodType(wrapped_prepare, target)
    patch = _Patch(spec, target, is_class_level, original_forward, original_prepare)
    _INSTALLED[id(target)] = patch
    return patch


def _uninstall(patch: _Patch) -> None:
    target = patch.target
    if patch.is_class_level:
        target.forward = patch.original_forward
        target._prepare_cache_for_generation = patch.original_prepare
    else:
        for name in ("forward", "_prepare_cache_for_generation"):
            if hasattr(target, name):
                delattr(target, name)  # fall back to the class method
    _INSTALLED.pop(id(target), None)


def _same_spec(a: FibQuantSpec, b: FibQuantSpec) -> bool:
    """Same operating point *and* same tensors (two independently rebuilt
    codebooks with identical shape are different specs)."""
    return (
        (a.d, a.k, a.n_levels, a.seed) == (b.d, b.k, b.n_levels, b.seed)
        and torch.equal(a.codebook, b.codebook)
        and torch.equal(a.rotation, b.rotation)
    )


def active_specs() -> dict[str, FibQuantSpec]:
    """Introspection: each patched target -> its active spec."""
    out: dict[str, FibQuantSpec] = {}
    for patch in _INSTALLED.values():
        name = getattr(patch.target, "__name__", type(patch.target).__name__)
        out[name] = patch.spec
    return out


class FibQuantRuntime:
    """One install lifecycle bound to an operating point.

    Construct with the spec (and optionally an explicit target model class for
    tests), then install/uninstall. A runtime for the class-level patch can be
    re-installed with a different spec through another runtime: the shared
    registry re-patches and warns instead of silently keeping the old spec.
    """

    def __init__(self, spec: FibQuantSpec, model_class: type | None = None):
        self.spec = spec
        self._model_class = model_class

    def install(self, model: torch.nn.Module | None = None) -> "FibQuantRuntime":
        """Patch forward() + the generate cache factory.

        model=None patches the class (class-level, covers lm-eval and future
        instances); passing a model patches that instance only.
        """
        if model is not None:
            target = model
            is_class_level = False
        else:
            target = self._model_class if self._model_class is not None else _default_model_class()
            is_class_level = True

        key = id(target)
        prev = _INSTALLED.get(key)
        if prev is not None:
            if _same_spec(prev.spec, self.spec):
                return self  # idempotent: same operating point already active
            logger.warning(
                "FibQuant re-install: replacing active spec (d=%d k=%d N=%d) with "
                "d=%d k=%d N=%d",
                prev.spec.d,
                prev.spec.k,
                prev.spec.n_levels,
                self.spec.d,
                self.spec.k,
                self.spec.n_levels,
            )
            _uninstall(prev)
        _install(target, self.spec, is_class_level)
        return self

    def uninstall(self, model: torch.nn.Module | None = None) -> None:
        """Restore the original methods for the (default) target."""
        target = model if model is not None else (self._model_class or _default_model_class())
        patch = _INSTALLED.get(id(target))
        if patch is not None:
            _uninstall(patch)

    @property
    def active_spec(self) -> FibQuantSpec | None:
        """The spec currently in effect for this runtime's target, or None."""
        target = self._model_class or _default_model_class()
        patch = _INSTALLED.get(id(target))
        return patch.spec if patch is not None else None
