"""The tool/plugin abstraction layer.

This package contains:

* :mod:`base`       — the ``ITool`` contract and the ``BaseTool`` helper.
* :mod:`registry`   — the ``ToolRegistry`` + ``create_default_registry`` factory.
* one module per controller domain (keyboard, mouse, window, filesystem,
  process, system), each contributing the concrete ``*Tool`` plugins.

New capabilities are added by writing one ``*Tool`` class and registering it in
:func:`~zoya.automation.tools.registry.create_default_registry` — nothing else
in the codebase needs to change (Open/Closed Principle).
"""

from zoya.automation.tools.base import BaseTool, ITool
from zoya.automation.tools.registry import ToolRegistry, create_default_registry

__all__ = ["BaseTool", "ITool", "ToolRegistry", "create_default_registry"]
