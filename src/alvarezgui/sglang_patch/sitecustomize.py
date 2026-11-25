"""Loaded by every sglang process (on PYTHONPATH, so spawned children load it too).

The MiniCPMV wrapper is a plain object that forwards attribute reads to the inner nn.Module, so the
loader's ``model.shared_experts_config = ...`` lands on the wrapper while runners read it from the inner
module. Mirror that write onto the inner module. Depends on sglang internals: re-probe before upgrading.
"""
import importlib.abc
import importlib.metadata
import importlib.util
import sys

PATCHED_VERSION = "0.5.23"


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "sglang.srt.models.minicpmv":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)

            def _setattr(self, attr, value):
                object.__setattr__(self, attr, value)
                if attr == "shared_experts_config" and self.__dict__.get("minicpmv") is not None:
                    setattr(self.__dict__["minicpmv"], attr, value)
            module.MiniCPMV.__setattr__ = _setattr
        spec.loader.exec_module = patched
        return spec


def install() -> bool:
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != PATCHED_VERSION:
        print(f"alvarez sglang patch: sglang {version} is not {PATCHED_VERSION}; not patched", file=sys.stderr)
        return False
    sys.meta_path.insert(0, _Hook())
    return True


install()
