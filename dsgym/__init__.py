"""
DSGym: Unified Data Science Benchmark Framework

A comprehensive platform for data science task evaluation and training.
"""

__version__ = "0.1.0"

# NOTE: submodules and their commonly-used classes are imported LAZILY (PEP 562
# module __getattr__) rather than eagerly. Eager `from . import datasets/agents/
# eval` pulled in the full pandas dataset loaders, the ReAct agent, and the
# evaluator on ANY `import dsgym.*` — which broke lightweight consumers that only
# need a narrow subtree (e.g. the in-container inline-PRM, which imports
# dsgym.adapra.{schema,llm_judge_v2,rubrics} and has no dataset deps). Lazy
# access preserves the public API (`dsgym.datasets`, `from dsgym import
# DatasetRegistry`, etc.) while only importing what's actually touched.

_LAZY_SUBMODULES = ("datasets", "agents", "eval")
_LAZY_ATTRS = {
    "DatasetRegistry": ("datasets", "DatasetRegistry"),
    "ReActDSAgent": ("agents", "ReActDSAgent"),
    "Evaluator": ("eval", "Evaluator"),
}

__all__ = [
    "datasets", "agents", "eval",
    "DatasetRegistry", "ReActDSAgent", "Evaluator",
    "__version__",
]


def __getattr__(name: str):
    """Lazily import submodules / commonly-used classes on first access."""
    import importlib
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        submod, attr = _LAZY_ATTRS[name]
        obj = getattr(importlib.import_module(f".{submod}", __name__), attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
