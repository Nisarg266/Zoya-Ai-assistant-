"""File-system operations built on :mod:`pathlib` and :mod:`shutil`.

Responsibility (SRP): *files & directories only*. Deliberately dependency-free
and cross-platform so this controller can be unit-tested anywhere and reused by
future modules (notes, memory cache, plugin assets).

Destructive operations log at WARNING level for an audit trail and, when
``safe_delete`` is enabled and :mod:`send2trash` is installed, prefer sending
items to the recycle bin instead of deleting them permanently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from zoya.core.exceptions import FileSystemError
from zoya.core.logging import get_logger

logger = get_logger("automation.filesystem")

# Anything that can be coerced into a pathlib.Path.
PathLike = Union[str, Path]


@dataclass(frozen=True)
class FileInfo:
    """Read-only metadata about a file or directory."""

    path: str
    name: str
    is_dir: bool
    size: int  # bytes
    modified: datetime


class FileSystemOperator:
    """Stateless wrapper around pathlib / shutil.

    Methods are synchronous and fast enough that they do not need to be wrapped
    in ``asyncio.to_thread`` for small files; the tool layer still offloads
    them for consistency with the rest of the subsystem.
    """

    def __init__(self, safe_delete: bool = True) -> None:
        self._safe_delete = safe_delete
        # Lazy, optional dependency — store the callable if available.
        self._send2trash: Optional[callable] = None  # type: ignore[type-arg]
        try:
            from send2trash import send2trash  # type: ignore

            self._send2trash = send2trash
        except Exception:
            # send2trash is optional; we just lose recycle-bin support.
            self._send2trash = None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _p(path: PathLike) -> Path:
        """Normalise input to an absolute, expanded :class:`Path`."""
        return Path(path).expanduser()

    # ------------------------------------------------------------------ #
    # Read-only operations                                                #
    # ------------------------------------------------------------------ #
    def exists(self, path: PathLike) -> bool:
        return self._p(path).exists()

    def list_directory(self, path: PathLike, pattern: str = "*") -> List[Path]:
        """List entries in ``path`` matching ``pattern`` (a glob)."""
        p = self._p(path)
        if not p.exists():
            raise FileSystemError(f"Directory not found: {p}")
        if not p.is_dir():
            raise FileSystemError(f"Not a directory: {p}")
        return sorted(p.glob(pattern))

    def search(self, directory: PathLike, pattern: str, recursive: bool = True) -> List[Path]:
        """Find paths under ``directory`` matching ``pattern``.

        ``recursive=True`` (default) uses :meth:`Path.rglob`, which supports
        ``**`` patterns such as ``"**/*.py"``.
        """
        p = self._p(directory)
        if not p.is_dir():
            raise FileSystemError(f"Directory not found: {p}")
        globber = p.rglob if recursive else p.glob
        return sorted(globber(pattern))

    def read_file(self, path: PathLike, encoding: str = "utf-8") -> str:
        """Read a text file's full contents."""
        p = self._p(path)
        if not p.is_file():
            raise FileSystemError(f"File not found: {p}")
        return p.read_text(encoding=encoding)

    def get_info(self, path: PathLike) -> FileInfo:
        """Return :class:`FileInfo` metadata for a path."""
        p = self._p(path)
        if not p.exists():
            raise FileSystemError(f"Path not found: {p}")
        stat = p.stat()
        return FileInfo(
            path=str(p),
            name=p.name,
            is_dir=p.is_dir(),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
        )

    # ------------------------------------------------------------------ #
    # Mutating operations                                                 #
    # ------------------------------------------------------------------ #
    def create_directory(self, path: PathLike, exist_ok: bool = True) -> Path:
        """Create a directory (and parents)."""
        p = self._p(path)
        p.mkdir(parents=True, exist_ok=exist_ok)
        return p

    def write_file(
        self, path: PathLike, content: str, append: bool = False, encoding: str = "utf-8"
    ) -> Path:
        """Write ``content`` to ``path`` (creating parent dirs as needed).

        ``append=True`` adds to the file instead of replacing it.
        """
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding=encoding) as fh:
            fh.write(content)
        return p

    def move(self, src: PathLike, dst: PathLike) -> Path:
        """Move a file or directory to ``dst``."""
        s, d = self._p(src), self._p(dst)
        if not s.exists():
            raise FileSystemError(f"Source not found: {s}")
        d.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(s), str(d))
        return d

    def copy(self, src: PathLike, dst: PathLike) -> Path:
        """Copy a file or directory tree to ``dst``."""
        s, d = self._p(src), self._p(dst)
        if not s.exists():
            raise FileSystemError(f"Source not found: {s}")
        d.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if s.is_dir():
            shutil.copytree(str(s), str(d), dirs_exist_ok=True)
        else:
            shutil.copy2(str(s), str(d))
        return d

    def delete(
        self,
        path: PathLike,
        recursive: bool = False,
        use_trash: Optional[bool] = None,
    ) -> None:
        """Delete a file or directory.

        Parameters
        ----------
        recursive:
            Required to delete a non-empty directory.
        use_trash:
            Override the controller default. When ``True`` (and
            :mod:`send2trash` is available) the item is sent to the recycle
            bin; otherwise it is deleted permanently.
        """
        p = self._p(path)
        if not p.exists():
            raise FileSystemError(f"Path not found: {p}")

        to_trash = self._safe_delete if use_trash is None else use_trash
        logger.warning("Deleting %s (recursive=%s, trash=%s)", p, recursive, to_trash)

        if to_trash and self._send2trash is not None:
            self._send2trash(str(p))
            return

        import shutil

        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()  # fails on non-empty dir — intentional safety
        else:
            p.unlink()


__all__ = ["FileInfo", "FileSystemOperator", "PathLike"]
