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
  - used as a context manager (`with FibQuantRuntime(spec): ...`), `install()`
    is scoped to the `with` block: on exit (even if the block raised) it
    restores exactly whatever was installed on the target *before* the block
    -- nothing, this same spec, or a different preexisting spec -- rather than
    unconditionally uninstalling, so nesting a scoped install inside an
    already-active one never deletes or permanently discards that outer
    state. This is what eval_harness.run_eval uses so a failed or repeated
    run never leaves a stale spec installed for the next one to inherit.
  - `active_spec` / `active_specs()` make the current state inspectable;
    class-level and instance-level installs on the same class are tracked (and
    reported) independently, so nesting one inside the other cannot hide a
    still-active spec.
  - `enable_fibquant(model=None, spec=None)`: the backwards-compatible
    (model, spec) wrapper around `FibQuantRuntime(spec).install(model=model)`,
    kept here rather than cache.py so cache.py (pure KV storage:
    FibQuantCache/FibQuantLayer) never has to import this module.

Both patch points are installed and removed together, so there is no
half-enabled state where forward() is compressed but generate() is not:

  - the model class's `forward()`: builds a FibQuantCache when use_cache is
    set and no cache was passed (plain forward calls, lm-eval loglikelihood);
  - the model class's `_prepare_cache_for_generation()` (the cache factory
    generate() calls in transformers 5.x): replaces the plain DynamicCache it
    just built with a FibQuantCache. User-supplied caches, static/quantized/
    offloaded implementations, and assisted generation are left untouched.

`FibQuantRuntime` is the single cache adapter: eval_harness.py never builds a
FibQuantCache itself, it only installs/uninstalls this runtime around the
lm-eval call so both the loglikelihood and generate() paths compress through
the same mechanism. A target class with no `_prepare_cache_for_generation`
(not GenerationMixin-capable, e.g. a plain decoder stack) can only have
forward() covered; `_install` logs that gap explicitly rather than raising or
silently pretending generate() is covered too.

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

__all__ = ["FibQuantRuntime", "active_specs", "enable_fibquant"]

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
    """Resolve lazily: the class-level install target when eval_harness.py
    calls `FibQuantRuntime(spec).install()` with no model instance yet.

    lm-eval's HFLM (backend="default", no override -- what run_eval() uses)
    picks AutoModelForCausalLM for a `qwen3_5` config
    (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES), landing on Qwen3_5ForCausalLM: the
    class whose forward() *and* _prepare_cache_for_generation() (inherited
    from GenerationMixin) actually run for both the loglikelihood and
    generate_until request paths run_eval() exercises. Qwen3_5TextModel (the
    plain decoder stack) has neither generate() nor a cache factory to patch,
    so a class-level install targeting it can cover forward() only -- not the
    generate() path lm-eval's HFLM._model_generate drives.
    """
    global _DEFAULT_MODEL_CLASS
    if _DEFAULT_MODEL_CLASS is None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        _DEFAULT_MODEL_CLASS = Qwen3_5ForCausalLM
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
    # An instance-level install must wrap the TRUE pristine methods, never an
    # already-active class-level patch's wrapped ones. Composing the two
    # (instance wrapper calling through to the class wrapper as its
    # "original") used to silently let the class-level spec win on the
    # generate() path: wrapped_prepare calls its original *first* to build
    # the cache, then only replaces it if it is still exactly a bare
    # DynamicCache; when the "original" was itself the class wrapper, that
    # inner call already swapped in the class spec's FibQuantCache, so the
    # instance wrapper's own type(cache) is DynamicCache check failed and it
    # silently kept the class spec instead of its own (forward() masked this
    # by accident -- its check-before-delegate order happens to let whichever
    # wrapper runs first win). A currently-active class-level patch's own
    # `original_forward`/`original_prepare` are always the true pristine
    # methods (a class-level re-install fully uninstalls -- restoring them --
    # before recapturing), so reusing those here whenever one is active makes
    # the instance-level wrapper call straight through to the real model
    # code: the instance's spec is the only one that ever runs for that
    # instance, and the instance attribute simply shadowing the class
    # attribute means uninstalling it (delattr) naturally falls back to
    # whatever the class currently provides -- patched or not.
    class_patch = None if is_class_level else _INSTALLED.get(id(cls))
    if class_patch is not None:
        original_forward = class_patch.original_forward
        original_prepare = class_patch.original_prepare
    else:
        original_forward = cls.forward
        # Not every patch target is generation-capable (e.g. a plain decoder
        # stack with no generate()/GenerationMixin): getattr instead of a bare
        # attribute access so install() covers forward() for it instead of
        # raising AttributeError -- the external-library seam this can't cover
        # (see _default_model_class) stays a logged, explicit no-op rather than a
        # crash or a silent skip.
        original_prepare = getattr(cls, "_prepare_cache_for_generation", None)
    wrapped_forward, wrapped_prepare = _make_wrappers(original_forward, original_prepare, spec)
    if is_class_level:
        target.forward = wrapped_forward
        if original_prepare is not None:
            target._prepare_cache_for_generation = wrapped_prepare
    else:  # instance attributes shadow the class methods
        target.forward = types.MethodType(wrapped_forward, target)
        if original_prepare is not None:
            target._prepare_cache_for_generation = types.MethodType(wrapped_prepare, target)
    if original_prepare is None:
        logger.info(
            "%s has no _prepare_cache_for_generation (not GenerationMixin-capable); "
            "FibQuant covers forward() only for this target -- generate() cache "
            "injection is out of scope here",
            getattr(cls, "__name__", cls),
        )
    patch = _Patch(spec, target, is_class_level, original_forward, original_prepare)
    _INSTALLED[id(target)] = patch
    return patch


def _uninstall(patch: _Patch) -> None:
    target = patch.target
    if patch.is_class_level:
        target.forward = patch.original_forward
        if patch.original_prepare is not None:
            target._prepare_cache_for_generation = patch.original_prepare
    else:
        names = ["forward"]
        if patch.original_prepare is not None:
            names.append("_prepare_cache_for_generation")
        for name in names:
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
    """Introspection: each patched target -> its active spec.

    Class-level and instance-level installs are independent entries (see
    _INSTALLED's id(target) keying above) and must stay independently visible
    here too: an instance-level patch on class C used to share C's plain
    name with a coexisting class-level patch on C, silently shadowing one
    spec with the other in the returned mapping -- a stale view of what is
    actually active on which target.
    """
    out: dict[str, FibQuantSpec] = {}
    for patch in _INSTALLED.values():
        if patch.is_class_level:
            name = getattr(patch.target, "__name__", str(patch.target))
        else:
            name = f"{type(patch.target).__name__} instance @ {id(patch.target):#x}"
        out[name] = patch.spec
    return out


class FibQuantRuntime:
    """One install lifecycle bound to an operating point.

    Construct with the spec (and optionally an explicit target model class for
    tests), then install/uninstall. A runtime for the class-level patch can be
    re-installed with a different spec through another runtime: the shared
    registry re-patches and warns instead of silently keeping the old spec.

    Used as a context manager (`with FibQuantRuntime(spec): ...`), it restores
    whatever was installed on the target *before* the block -- nothing, this
    same spec, or a different spec -- rather than unconditionally uninstalling
    on exit; see `__enter__`/`__exit__`.
    """

    def __init__(self, spec: FibQuantSpec, model_class: type | None = None):
        self.spec = spec
        self._model_class = model_class
        self._context_target: Any = None
        self._prior_patch: _Patch | None = None

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

    def __enter__(self) -> "FibQuantRuntime":
        """Class-level install scoped to a `with` block.

        Callers (run_eval, in particular) that need every patch undone when
        the block exits -- including when it raises -- get that for free
        instead of pairing install()/uninstall() by hand around a
        try/finally. Whatever was already installed on this target *before*
        the block -- nothing, the same spec, or a different spec -- is
        snapshotted here so `__exit__` can put it back exactly, rather than
        always tearing down to the pristine unpatched class: a bare
        `self.uninstall()` on exit would delete a same-spec patch that
        predates this `with`, or leave a *different* preexisting spec
        replaced by this block's spec permanently discarded.
        """
        self._context_target = self._model_class if self._model_class is not None else _default_model_class()
        self._prior_patch = _INSTALLED.get(id(self._context_target))
        self.install()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore exactly what was installed on this target before `__enter__`.

        Three cases, keyed off the snapshot taken in `__enter__`:
          - nothing was installed before: fully uninstall (back to the
            pristine class), same as a plain `uninstall()`.
          - the same spec was already active: `install()` took its idempotent
            path and never touched `_INSTALLED`, so the prior patch is still
            exactly in place -- leave it alone.
          - a *different* spec was already active: `install()` uninstalled it
            to install this block's spec, so its true original methods are
            already restored; re-`_install()` the prior spec on top of them to
            reproduce that exact prior patched state.
        """
        target = self._context_target
        prior = self._prior_patch
        self._context_target = None
        self._prior_patch = None
        if target is None:
            return
        key = id(target)
        current = _INSTALLED.get(key)
        if prior is None:
            self.uninstall()
            return
        if current is prior:
            return  # install() was a no-op; the prior patch was never touched
        if current is not None:
            _uninstall(current)
        _install(target, prior.spec, prior.is_class_level)


def enable_fibquant(
    model: torch.nn.Module | None = None,
    spec: FibQuantSpec | None = None,
) -> FibQuantRuntime:
    """Backwards-compatible install wrapper; prefer FibQuantRuntime.

    Preserves the old (model, spec) call shape. Unlike the original one-shot
    patch, the runtime install is idempotent per operating point, re-patches
    (with a warning) when a different spec is installed, covers generate()
    via the cache-factory patch, and offers uninstall()/active_spec. Lives
    here (not cache.py) so cache.py -- FibQuantCache/FibQuantLayer, pure KV
    storage -- never has to import runtime.py: the only cross-module edge is
    runtime.py's own lazy `from .cache import FibQuantCache` inside
    `_new_cache`, used solely at the moment a cache is actually constructed.
    """
    if spec is None:
        raise ValueError("enable_fibquant requires a spec (pass spec=...)")
    return FibQuantRuntime(spec).install(model=model)
