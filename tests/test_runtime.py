"""FibQuantRuntime lifecycle: scoping, idempotency, re-patch, uninstall, generate coverage.

The dummy model class stands in for Qwen3_5TextModel: the installer takes an
injectable class, and the patch contract (forward + _prepare_cache_for_generation)
is exactly what it touches — so these tests need no model and no GPU.
"""

import logging
import types

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.configuration_utils import GenerationConfig

from fibquant.cache import FibQuantCache
from fibquant.runtime import FibQuantRuntime, active_specs


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
