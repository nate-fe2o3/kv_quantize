"""FibQuantRuntime lifecycle: scoping, idempotency, re-patch, uninstall, generate coverage.

The dummy model class stands in for a generation-capable Qwen3.5 class (e.g.
Qwen3_5ForCausalLM, the real default install target -- see
runtime._default_model_class): the installer takes an injectable class, and
the patch contract (forward + _prepare_cache_for_generation) is exactly what
it touches — so these tests need no model and no GPU.
"""

import logging
import types

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.configuration_utils import GenerationConfig

from fibquant.cache import FibQuantCache
from fibquant.runtime import FibQuantRuntime, active_specs, enable_fibquant


class DummyConfig:
    def __init__(self):
        self.layer_types = ["full_attention"]

    def get_text_config(self, decoder=False):
        return self


class DummyModel:
    """Mirrors the Qwen3.5 methods the runtime patches."""

    def __init__(self):
        self.config = DummyConfig()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        # Returns (cache, input_ids) so tests can verify positional arguments
        # survive the wrapper's call (a double-bound method shifts them).
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        return past_key_values, input_ids

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                      generation_mode=None, batch_size=1, max_cache_length=0):
        # mirrors transformers 5.15.1: the default cache is a plain DynamicCache
        model_kwargs["past_key_values"] = DynamicCache()
        return True


class DummyTextOnlyModel:
    """Mirrors a plain decoder stack with no generate()/GenerationMixin (e.g.
    Qwen3_5TextModel): forward() only, no `_prepare_cache_for_generation`."""

    def __init__(self):
        self.config = DummyConfig()

    def forward(self, input_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        return past_key_values, input_ids


@pytest.fixture(autouse=True)
def clean_installations():
    """No install may leak between tests."""
    yield
    specs = active_specs()
    if "DummyModel" in specs:
        FibQuantRuntime(specs["DummyModel"], model_class=DummyModel).uninstall()



def test_class_install_builds_fibquant_cache_on_forward(small_spec):
    original = DummyModel.forward
    FibQuantRuntime(small_spec, model_class=DummyModel).install()
    assert DummyModel.forward is not original
    model = DummyModel()
    ids = torch.tensor([1, 2])
    cache, got_ids = model.forward(input_ids=ids, use_cache=True)
    assert got_ids is ids  # argument not shifted by the wrapper
    assert isinstance(cache, FibQuantCache)
    assert cache.spec.n_levels == small_spec.n_levels


def test_forward_untouched_without_use_cache_or_with_explicit_cache(small_spec):
    FibQuantRuntime(small_spec, model_class=DummyModel).install()
    model = DummyModel()
    cache, _ = model.forward(use_cache=False)
    assert cache is None
    mine = DynamicCache()
    cache, _ = model.forward(use_cache=True, past_key_values=mine)
    assert cache is mine


def test_generate_cache_factory_replaces_dynamic_cache(small_spec):
    FibQuantRuntime(small_spec, model_class=DummyModel).install()
    model = DummyModel()
    kwargs = {}
    model._prepare_cache_for_generation(GenerationConfig(), kwargs)
    assert type(kwargs["past_key_values"]) is FibQuantCache
    assert kwargs["past_key_values"].spec.n_levels == small_spec.n_levels


def test_generate_cache_factory_restored_after_uninstall(small_spec):
    runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    runtime.install()
    runtime.uninstall()
    model = DummyModel()
    kwargs = {}
    model._prepare_cache_for_generation(GenerationConfig(), kwargs)
    assert type(kwargs["past_key_values"]) is DynamicCache


def test_install_is_idempotent_for_same_spec(small_spec):
    runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    runtime.install()
    wrapped = DummyModel.forward
    runtime.install()
    assert DummyModel.forward is wrapped  # no double wrapping


def test_reinstall_with_different_spec_warns_and_repairs(small_spec, packed_spec, caplog):
    FibQuantRuntime(small_spec, model_class=DummyModel).install()
    with caplog.at_level(logging.WARNING, logger="fibquant.runtime"):
        FibQuantRuntime(packed_spec, model_class=DummyModel).install()
    assert any("replacing active spec" in r.message for r in caplog.records)
    model = DummyModel()
    cache, _ = model.forward(use_cache=True)
    assert cache.spec.n_levels == packed_spec.n_levels


def test_uninstall_restores_original_methods(small_spec):
    original_forward = DummyModel.forward
    original_prepare = DummyModel._prepare_cache_for_generation
    runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    runtime.install()
    runtime.uninstall()
    assert DummyModel.forward is original_forward
    assert DummyModel._prepare_cache_for_generation is original_prepare
    assert active_specs() == {}


def test_instance_install_is_scoped_to_that_model(small_spec):
    runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    m1, m2 = DummyModel(), DummyModel()
    runtime.install(model=m1)
    assert isinstance(m1.forward, types.MethodType)  # bound wrapper on the instance
    ids = torch.tensor([3, 4])
    cache, got_ids = m1.forward(input_ids=ids, use_cache=True)
    assert got_ids is ids  # instance binding did not shift the argument
    assert isinstance(cache, FibQuantCache)
    cache2, got2 = m2.forward(input_ids=ids, use_cache=True)
    assert isinstance(cache2, DynamicCache) and not isinstance(cache2, FibQuantCache)
    assert got2 is ids  # other instances keep the original class method
    kwargs = {}
    m1._prepare_cache_for_generation(GenerationConfig(), kwargs)
    assert type(kwargs["past_key_values"]) is FibQuantCache
    runtime.uninstall(model=m1)
    # instance attr gone; the class method is shadowed no more
    assert m1.forward.__func__ is DummyModel.forward
    cache3, _ = m1.forward(use_cache=True)
    assert not isinstance(cache3, FibQuantCache)


def test_active_spec_reflects_state(small_spec):
    runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    assert runtime.active_spec is None
    runtime.install()
    assert runtime.active_spec is not None and runtime.active_spec.n_levels == 16
    assert active_specs() == {"DummyModel": runtime.active_spec}
    runtime.uninstall()
    assert runtime.active_spec is None


def test_context_manager_installs_and_uninstalls(small_spec):
    original_forward = DummyModel.forward
    original_prepare = DummyModel._prepare_cache_for_generation
    with FibQuantRuntime(small_spec, model_class=DummyModel) as runtime:
        assert DummyModel.forward is not original_forward
        assert runtime.active_spec is not None and runtime.active_spec.n_levels == 16
        model = DummyModel()
        cache, _ = model.forward(use_cache=True)
        assert isinstance(cache, FibQuantCache)
    # __exit__ uninstalled even though nothing raised
    assert DummyModel.forward is original_forward
    assert DummyModel._prepare_cache_for_generation is original_prepare
    assert active_specs() == {}


def test_context_manager_uninstalls_on_exception(small_spec):
    original_forward = DummyModel.forward
    with pytest.raises(RuntimeError, match="boom"):
        with FibQuantRuntime(small_spec, model_class=DummyModel):
            assert DummyModel.forward is not original_forward
            raise RuntimeError("boom")
    # __exit__ still restored everything despite the exception
    assert DummyModel.forward is original_forward
    assert active_specs() == {}


def test_context_manager_with_same_spec_preinstalled_leaves_it_active(small_spec):
    """A `with FibQuantRuntime(spec)` nested inside an already-active install
    of that *same* spec must not tear that outer install down on exit --
    install() takes its idempotent no-op path, so the preexisting patch was
    never touched and __exit__ must leave it exactly as it found it."""
    outer = FibQuantRuntime(small_spec, model_class=DummyModel)
    outer.install()
    try:
        forward_during_outer = DummyModel.forward
        with FibQuantRuntime(small_spec, model_class=DummyModel) as inner:
            assert DummyModel.forward is forward_during_outer  # idempotent: unchanged
            assert inner.active_spec is not None and inner.active_spec.n_levels == small_spec.n_levels
        # exiting the inner scope must NOT have uninstalled the still-active outer spec
        assert DummyModel.forward is forward_during_outer
        assert outer.active_spec is not None
        assert active_specs() == {"DummyModel": outer.active_spec}
    finally:
        outer.uninstall()
    assert active_specs() == {}


def test_context_manager_with_different_spec_restores_previous_spec_after(small_spec, packed_spec):
    """A `with FibQuantRuntime(other_spec)` nested inside an already-active
    install of a *different* spec must restore that outer spec on exit, not
    leave the class unpatched and not leave the inner spec installed."""
    outer = FibQuantRuntime(small_spec, model_class=DummyModel)
    outer.install()
    try:
        with FibQuantRuntime(packed_spec, model_class=DummyModel) as inner:
            assert inner.active_spec is not None and inner.active_spec.n_levels == packed_spec.n_levels
            model = DummyModel()
            cache, _ = model.forward(use_cache=True)
            assert cache.spec.n_levels == packed_spec.n_levels  # inner spec active during the block
        # exiting must restore the outer spec exactly, not the pristine class
        assert active_specs()["DummyModel"].n_levels == small_spec.n_levels
        model = DummyModel()
        cache, _ = model.forward(use_cache=True)
        assert cache.spec.n_levels == small_spec.n_levels  # back to the outer spec, not packed_spec
        assert active_specs()["DummyModel"].n_levels == small_spec.n_levels
    finally:
        outer.uninstall()
    assert active_specs() == {}


def test_install_skips_missing_prepare_cache_hook_gracefully(small_spec, caplog):
    """Regression: a target with no _prepare_cache_for_generation (not
    GenerationMixin-capable, e.g. a plain decoder stack) must not crash
    install() -- forward() coverage still applies, and the gap is logged
    explicitly rather than silently ignored."""
    assert not hasattr(DummyTextOnlyModel, "_prepare_cache_for_generation")
    runtime = FibQuantRuntime(small_spec, model_class=DummyTextOnlyModel)
    try:
        with caplog.at_level(logging.INFO, logger="fibquant.runtime"):
            runtime.install()
        assert any(
            "has no _prepare_cache_for_generation" in r.message for r in caplog.records
        )
        model = DummyTextOnlyModel()
        cache, _ = model.forward(use_cache=True)
        assert isinstance(cache, FibQuantCache)
        assert not hasattr(model, "_prepare_cache_for_generation")
        assert runtime.active_spec is not None
    finally:
        runtime.uninstall()
    assert runtime.active_spec is None


def test_default_target_class_install_uninstall_does_not_crash(small_spec):
    """Regression: the real default class (see runtime._default_model_class)
    previously had no `_prepare_cache_for_generation`, so a bare
    `FibQuantRuntime(spec).install()` -- exactly eval_harness.run_eval's
    spec-enabled path -- raised AttributeError on every real invocation.
    """
    runtime = FibQuantRuntime(small_spec)
    try:
        runtime.install()
        assert runtime.active_spec is not None
        assert runtime.active_spec.n_levels == small_spec.n_levels
    finally:
        runtime.uninstall()
    assert runtime.active_spec is None


def test_class_and_instance_installs_are_both_visible_in_active_specs(small_spec, packed_spec):
    """Nesting a class-level install with an instance-level install on an
    instance of that same class must not have one hide the other behind a
    shared name in active_specs() -- each target's spec must stay visible.

    Also a behavioral regression: the instance-level spec must be the one
    that actually wins on the generate() cache-factory path
    (_prepare_cache_for_generation), not just on forward(). The instance
    wrapper used to close over the class wrapper as its "original", so by
    the time it checked whether to inject its own FibQuantCache, the class
    wrapper had already injected *its* spec's cache and the check silently
    no-opped -- forward() masked this by accident (its check runs before
    delegating), _prepare_cache_for_generation did not (it delegates first).
    """
    class_runtime = FibQuantRuntime(small_spec, model_class=DummyModel)
    class_runtime.install()
    instance = DummyModel()
    instance_runtime = FibQuantRuntime(packed_spec, model_class=DummyModel)
    instance_runtime.install(model=instance)
    try:
        specs = active_specs()
        assert len(specs) == 2
        assert specs["DummyModel"] is small_spec
        instance_key = next(k for k in specs if k != "DummyModel")
        assert "instance" in instance_key
        assert specs[instance_key] is packed_spec
        # each target still resolves its own spec through forward()/generate()
        other = DummyModel()
        cache, _ = other.forward(use_cache=True)
        assert cache.spec.n_levels == small_spec.n_levels  # class-level spec

        cache, _ = instance.forward(use_cache=True)
        assert cache.spec.n_levels == packed_spec.n_levels  # instance-level spec

        # generate()'s cache factory: the instance spec must win here too,
        # not the class spec it's nested inside.
        other_kwargs: dict = {}
        other._prepare_cache_for_generation(GenerationConfig(), other_kwargs)
        assert type(other_kwargs["past_key_values"]) is FibQuantCache
        assert other_kwargs["past_key_values"].spec.n_levels == small_spec.n_levels

        instance_kwargs: dict = {}
        instance._prepare_cache_for_generation(GenerationConfig(), instance_kwargs)
        assert type(instance_kwargs["past_key_values"]) is FibQuantCache
        assert instance_kwargs["past_key_values"].spec.n_levels == packed_spec.n_levels
    finally:
        instance_runtime.uninstall(model=instance)
        # uninstalling the instance patch falls back to the still-active
        # class patch, not to the pristine unpatched method.
        assert instance.forward.__func__ is DummyModel.forward
        cache, _ = instance.forward(use_cache=True)
        assert cache.spec.n_levels == small_spec.n_levels
        kwargs: dict = {}
        instance._prepare_cache_for_generation(GenerationConfig(), kwargs)
        assert kwargs["past_key_values"].spec.n_levels == small_spec.n_levels
        class_runtime.uninstall()
    assert active_specs() == {}


def test_enable_fibquant_requires_a_spec():
    """Backwards-compatible wrapper: the (model, spec) call shape is
    preserved, but a spec is mandatory -- there is nothing to "enable"
    without one."""
    with pytest.raises(ValueError, match="requires a spec"):
        enable_fibquant(spec=None)


def test_enable_fibquant_installs_class_level_on_real_default_class(small_spec):
    """enable_fibquant(model=None, spec=...) must behave exactly like
    FibQuantRuntime(spec).install() against the real default target class
    (see runtime._default_model_class) -- this is the wrapper's whole
    purpose, so it has to actually delegate to FibQuantRuntime, not just
    resemble it."""
    runtime = enable_fibquant(spec=small_spec)
    try:
        assert isinstance(runtime, FibQuantRuntime)
        assert runtime.active_spec is small_spec
    finally:
        runtime.uninstall()
    assert runtime.active_spec is None


def test_enable_fibquant_installs_instance_level_when_model_given(small_spec):
    """enable_fibquant(model=an_instance, spec=...) must patch only that
    instance, exactly like FibQuantRuntime(spec).install(model=instance)."""
    model = DummyModel()
    runtime = enable_fibquant(model=model, spec=small_spec)
    try:
        assert isinstance(runtime, FibQuantRuntime)
        cache, _ = model.forward(use_cache=True)
        assert isinstance(cache, FibQuantCache)
        assert cache.spec.n_levels == small_spec.n_levels
        other = DummyModel()
        plain_cache, _ = other.forward(use_cache=True)
        assert not isinstance(plain_cache, FibQuantCache)  # scoped to `model` only
    finally:
        runtime.uninstall(model=model)


def test_cache_module_no_longer_imports_or_reexports_runtime():
    """Regression: cache.py (pure KV storage: FibQuantCache/FibQuantLayer)
    must not import runtime.py or re-export FibQuantRuntime/enable_fibquant
    -- that import/re-export cycle moved entirely into runtime.py, which
    already lazily imports FibQuantCache from cache.py only inside
    `_new_cache` (at the point a cache is actually constructed)."""
    import fibquant.cache as cache_mod

    assert not hasattr(cache_mod, "FibQuantRuntime")
    assert not hasattr(cache_mod, "enable_fibquant")
    assert "FibQuantRuntime" not in cache_mod.__all__
    assert "enable_fibquant" not in cache_mod.__all__
    assert "runtime" not in vars(cache_mod)
