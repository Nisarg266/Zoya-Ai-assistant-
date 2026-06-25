"""Tool registry for the Desktop Automation module.

Manages the collection of available tools and handles instantiation.
"""

from typing import Any

from .base import ITool


class ToolRegistry:
    """A registry that holds instantiated tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ITool] = {}

    def register(self, tool: ITool) -> None:
        """Register an instantiated tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ITool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[ITool]:
        """Return a list of all registered tools."""
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Get the JSON schema for all registered tools."""
        return [tool.schema() for tool in self._tools.values()]


def create_default_registry() -> ToolRegistry:
    """Factory to create a ToolRegistry pre-populated with all standard tools.

    This is the application's single **composition root**: the only place
    controllers are instantiated and injected into tools. Adding a capability is
    therefore a one-line change here (Open/Closed Principle) — nothing else in
    the codebase needs to change.

    Note
    ----
    Controller/tool imports are local so that ``import zoya.automation.tools``
    stays lightweight and cycle-free; the concrete tools are only needed when a
    registry is actually built.
    """
    registry = ToolRegistry()

    # --- Application launcher -------------------------------------------
    from zoya.automation.controllers.app_controller import AppController
    from zoya.automation.tools.app import OpenAppTool

    app_controller = AppController()
    registry.register(OpenAppTool(app_controller))

    # Future tools (keyboard, mouse, window, filesystem, process, system)
    # plug in here, each as: registry.register(XxxTool(controller)).

    return registry
