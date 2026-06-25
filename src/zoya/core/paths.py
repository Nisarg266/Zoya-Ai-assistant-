"""Centralised path resolution for Zoya.

Why a dedicated module?
-----------------------
Hard-coding ``Path(__file__).parents[N]`` across the codebase is fragile — a
single moved file silently shifts the index and breaks every path. This module
computes the project root **once**, exposes it as a constant, and offers a
small, immutable :class:`ProjectPaths` object that resolves the runtime
directories Zoya needs (logs, screenshots, notes, config).

Everything is *resolved* (absolute + symlinks flattened) and created lazily on
demand via :meth:`ProjectPaths.ensure_dirs`, so callers never have to worry
about ``FileNotFoundError`` for well-known folders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Project root, derived from this file's location.
#:
#:   src/zoya/core/paths.py
#:     parents[0] -> core/
#:     parents[1] -> zoya/
#:     parents[2] -> src/
#:     parents[3] -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable bundle of well-known, resolved filesystem locations.

    ``base`` anchors every relative path supplied by configuration; pass the
    project root (the default) so that ``log_dir = "logs"`` resolves to
    ``<project>/logs``.
    """

    base: Path = field(default_factory=lambda: PROJECT_ROOT)

    # ---- derived, read-only properties ----------------------------------
    @property
    def config_file(self) -> Path:
        """Default YAML settings file (``config/settings.yaml``)."""
        return self.base / "config" / "settings.yaml"

    @property
    def env_file(self) -> Path:
        """The ``.env`` file at the project root."""
        return self.base / ".env"

    @property
    def log_dir(self) -> Path:
        return self.base / "logs"

    @property
    def screenshot_dir(self) -> Path:
        return self.base / "screenshots"

    @property
    def notes_dir(self) -> Path:
        return self.base / "data" / "notes"

    # ---- helpers --------------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        """Resolve an arbitrary path.

        Relative paths are anchored at :attr:`base`; absolute paths are
        returned unchanged (and normalised).
        """
        p = Path(path)
        return p if p.is_absolute() else (self.base / p).resolve()

    def ensure_dirs(self, *which: str) -> list[Path]:
        """Create the requested well-known directories.

        With no arguments, creates all runtime folders Zoya uses. Missing
        parent directories are created automatically. Idempotent.

        Returns the list of directories that were ensured (for logging).
        """
        registry: dict[str, Path] = {
            "logs": self.log_dir,
            "screenshots": self.screenshot_dir,
            "notes": self.notes_dir,
        }
        targets = which or tuple(registry)
        created: list[Path] = []
        for key in targets:
            try:
                target = registry[key]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Unknown runtime directory '{key}'. "
                    f"Known: {sorted(registry)}"
                ) from exc
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
        return created


#: Singleton instance used everywhere — cheap to construct, safe to share.
PATHS: ProjectPaths = ProjectPaths()


__all__ = ["PROJECT_ROOT", "ProjectPaths", "PATHS"]
