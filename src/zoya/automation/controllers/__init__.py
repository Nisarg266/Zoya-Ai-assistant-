"""Low-level controllers for desktop automation.

Each controller has a single responsibility (SRP) and exposes a small, focused,
synchronous API. They are intentionally platform-aware: Windows-only controllers
guard their imports so the package can still be imported (and partly unit-tested)
on other platforms — instantiating them off-Windows raises a clear
:class:`~zoya.core.exceptions.AutomationError`.
"""
