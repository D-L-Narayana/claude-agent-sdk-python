"""Export a :class:`SessionStore`-backed session to the local transcript layout.

This is the inverse of :mod:`session_import` and the durable sibling of
:mod:`session_resume`: where ``materialize_resume_session`` writes a store-held
session into a *temporary* ``CLAUDE_CONFIG_DIR`` that lives only as long as the
resumed subprocess, ``export_session_from_store`` writes it into the *real*
projects directory (``~/.claude/projects/`` or ``$CLAUDE_CONFIG_DIR/projects/``)
so the plain CLI can ``claude --resume <session_id>`` it from the project
directory and the disk-reading helpers (``list_sessions``,
``get_session_messages``, ``list_subagents``, ``get_subagent_messages``) can
see it.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

from ..types import SessionKey, SessionStore
from .session_resume import _with_timeout, _write_jsonl, _write_session_subkeys
from .session_store_validation import _store_implements
from .sessions import (
    _get_projects_dir,
    _type_first,
    _validate_uuid,
    project_key_for_directory,
)

__all__ = ["export_session_from_store"]

logger = logging.getLogger(__name__)

# Operation name used in ``_with_timeout`` errors raised by the export path.
_EXPORT_PHASE = "session export"


async def export_session_from_store(
    session_store: SessionStore,
    session_id: str,
    *,
    directory: str | None = None,
    include_subagents: bool = True,
    overwrite: bool = False,
    load_timeout_ms: int = 60_000,
) -> Path:
    """Write a session held in ``session_store`` to the local projects directory.

    Loads the session's entries (and, by default, its subagent transcripts)
    from the store and writes them as
    ``<projects_dir>/<project_key>/<session_id>.jsonl`` — one compact JSON
    object per line, mode ``0o600`` — plus
    ``<projects_dir>/<project_key>/<session_id>/subagents/.../agent-<id>.jsonl``
    and ``agent-<id>.meta.json`` sidecars, exactly where the CLI writes them.
    ``projects_dir`` is ``$CLAUDE_CONFIG_DIR/projects`` when that variable is
    set, else ``~/.claude/projects``; ``project_key`` is
    :func:`project_key_for_directory` ``(directory)``, the same key the live
    mirror and :func:`import_session_to_store` use, so a session mirrored from
    (or imported from) ``directory`` exports back to the place it came from.

    The result is readable by ``get_session_messages(session_id,
    directory=directory)``, ``list_sessions(directory=directory)``,
    ``list_subagents`` / ``get_subagent_messages``, and resumable by the plain
    CLI: run ``claude --resume <session_id>`` from ``directory`` (with the same
    ``CLAUDE_CONFIG_DIR``, if any). Unlike the resume materialization, no auth
    or settings files are touched — the real config directory already has them.

    Nothing is written at the target paths until every store call has
    completed: the files are staged in a temporary directory inside the
    project directory and only then published, subagent files first and the
    main transcript last, so a store timeout or adapter failure leaves no
    partial session behind and ``list_sessions`` / ``--resume`` never see a
    transcript whose subagent history is missing. With ``overwrite=False``
    each file is published by an exclusive create that can never replace an
    existing file — so a transcript the CLI or another process wrote at the
    same path *while this export was awaiting the store* is preserved and the
    export is refused with :class:`FileExistsError` instead. A refused or
    failed export undoes exactly what it created (its own files, and the
    directories it made if they are still empty) and nothing else; a project
    directory created only for a failed export is removed again.

    Args:
        session_store: The store to read from. Subagent transcripts are
            exported only if the store implements
            :meth:`SessionStore.list_subkeys`.
        session_id: UUID of the session to export.
        directory: Project directory the session belongs to (same semantics
            as :func:`list_sessions`). Defaults to the current working
            directory.
        include_subagents: If ``True`` (default), also export subagent
            transcripts under ``<session_id>/subagents/**`` and their
            ``.meta.json`` sidecars.
        overwrite: If ``False`` (default), never replace an existing file:
            the export is refused if ``<session_id>.jsonl`` — or any subagent
            transcript / sidecar it would write — already exists, whether it
            existed beforehand or appeared while the export was awaiting the
            store. If ``True``, replace the main transcript and (re)write
            every subagent transcript/sidecar the store holds; files already
            in the session directory that the store does not know about are
            left in place. Do not export over a session the CLI is actively
            writing.
        load_timeout_ms: Timeout for each store call (``load()`` and, when
            applicable, ``list_subkeys()``), in milliseconds. Default 60 s.

    Returns:
        Path of the written main transcript.

    Raises:
        ValueError: If ``session_id`` is not a valid UUID.
        FileExistsError: If ``overwrite`` is ``False`` and a file already
            occupies a path this export would write; the message names that
            path. A main transcript present before the export is refused by a
            cheap pre-check before the store is consulted; one that appears
            later (or an existing subagent file) is refused at publish time,
            with nothing of this export left behind.
        FileNotFoundError: If the store has no entries for the session under
            ``project_key``.
        RuntimeError: If a store call times out or raises; the message names
            the call (``SessionStore.load()`` / ``list_subkeys()``) and the
            session.

    Note:
        Subpaths returned by ``list_subkeys()`` that would escape the session
        directory (absolute, ``..``, drive-prefixed, empty) are skipped with a
        warning, mirroring the resume path. ``type`` is hoisted to the front
        of each exported object so ``list_sessions`` tag/title extraction,
        which keys on a ``{"type":"tag"`` line prefix, works even for adapters
        that reorder JSON keys (e.g. Postgres ``JSONB``).

    Note:
        The exclusive publish (``overwrite=False``) hard-links the staged file
        — already mode ``0o600`` — to its target, which fails atomically if the
        target exists. On filesystems that refuse hard links (some FAT/SMB
        mounts, restricted containers) it falls back to creating the target
        with ``O_CREAT | O_EXCL`` and mode ``0o600`` and copying the bytes; the
        guarantee is the same.

    Example:
        Make a store-held session resumable by the CLI from its project::

            path = await export_session_from_store(
                store, session_id, directory="/path/to/project"
            )
            # then: cd /path/to/project && claude --resume <session_id>
            messages = get_session_messages(session_id, directory="/path/to/project")
    """
    if _validate_uuid(session_id) is None:
        raise ValueError(f"Invalid session_id: {session_id}")
    timeout_s = load_timeout_ms / 1000
    project_key = project_key_for_directory(directory)
    project_dir = _get_projects_dir() / project_key
    target = project_dir / f"{session_id}.jsonl"
    # Cheap local precondition first, so a refused export never costs a
    # (possibly remote, possibly slow) store load. This is only a fast path:
    # the exclusive publish below is what refuses a target that appears while
    # the store is being awaited.
    if target.exists() and not overwrite:
        raise _already_exists(target)

    key: SessionKey = {"project_key": project_key, "session_id": session_id}
    entries = await _with_timeout(
        session_store.load(key),
        timeout_s,
        f"SessionStore.load() for session {session_id}",
        phase=_EXPORT_PHASE,
    )
    if not entries:
        raise FileNotFoundError(
            f"Session {session_id} not found in session_store "
            f"(project_key={project_key!r})"
        )

    project_dir_preexisting = project_dir.is_dir()
    project_dir.mkdir(parents=True, exist_ok=True)
    session_dir = project_dir / session_id
    # Stage inside the project directory (same filesystem) so publishing is a
    # hard link / rename, never a copy across devices. The dotted name keeps
    # the staging dir out of session listings, which only consider
    # ``<uuid>.jsonl`` files.
    staging = Path(
        tempfile.mkdtemp(prefix=".export-", suffix=f"-{session_id}", dir=project_dir)
    )
    publisher = _Publisher(overwrite=overwrite)
    try:
        staged_main = staging / target.name
        _write_jsonl(staged_main, [_type_first(e) for e in entries])

        staged_session_dir = staging / session_id
        if include_subagents and _store_implements(session_store, "list_subkeys"):
            await _write_session_subkeys(
                session_store,
                staged_session_dir,
                project_key,
                session_id,
                timeout_s,
                phase=_EXPORT_PHASE,
            )

        # Every store call has succeeded — publish the staged files, main
        # transcript last so readers never see it without its subagent files.
        for staged in _staged_files(staged_session_dir):
            dst = session_dir / staged.relative_to(staged_session_dir)
            publisher.ensure_dir(dst.parent, within=project_dir)
            publisher.publish(staged, dst)
        publisher.publish(staged_main, target)
    except BaseException:
        # Undo exactly what this export created — never a concurrent writer's
        # files — then re-raise. BaseException so the backend's cancellation
        # exception also triggers the cleanup.
        publisher.rollback()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not project_dir_preexisting and not target.exists():
            # The project directory was created for this export and the
            # export did not happen — remove it again. ``rmdir`` refuses a
            # non-empty directory, so anything else that landed there (e.g.
            # a concurrent CLI write) is left alone.
            with suppress(OSError):
                project_dir.rmdir()

    logger.debug("[SessionStore] exported session %s to %s", session_id, target)
    return target


def _already_exists(path: Path) -> FileExistsError:
    """The error an ``overwrite=False`` export raises for an occupied path."""
    return FileExistsError(f"{path} already exists; pass overwrite=True to replace it")


def _staged_files(staged_dir: Path) -> list[Path]:
    """Files under ``staged_dir`` in a deterministic order (empty if absent)."""
    if not staged_dir.is_dir():
        return []
    return sorted(p for p in staged_dir.rglob("*") if p.is_file())


def _hardlink(staged: Path, target: Path) -> None:
    """Create ``target`` as a hard link to ``staged``.

    An atomic create-if-absent: it fails with :class:`FileExistsError` when
    ``target`` already exists and never replaces anything. Module-level so a
    test can simulate a filesystem without hard links.
    """
    target.hardlink_to(staged)


def _identity(st: os.stat_result) -> tuple[int, int]:
    """``(inode, device)`` — identifies a file independently of its name."""
    return st.st_ino, st.st_dev


class _Publisher:
    """Publish staged files at their targets, remembering exactly what it created.

    With ``overwrite=False`` every file is published by an exclusive operation
    that can never replace an existing file: a hard link of the staged inode
    (which already carries mode 0o600) to the target or — where the filesystem
    refuses hard links for any reason other than "target exists" — an
    exclusive create (``O_CREAT | O_EXCL``, mode 0o600) of the staged bytes.
    With ``overwrite=True`` files are moved into place by an atomic rename that
    replaces an existing target.

    :meth:`rollback` removes only the files this publisher created — checked by
    inode identity, so a file someone else has replaced in the meantime is left
    alone — and only the directories it created that are still empty.
    """

    def __init__(self, *, overwrite: bool) -> None:
        self.overwrite = overwrite
        self._created_files: list[tuple[Path, tuple[int, int]]] = []
        self._created_dirs: list[Path] = []
        self._hardlinks_ok = True

    def ensure_dir(self, directory: Path, *, within: Path) -> None:
        """Create ``directory`` and its missing ancestors below ``within``
        (which must exist), recording the ones created here."""
        d = within
        for part in directory.relative_to(within).parts:
            d = d / part
            try:
                d.mkdir()
            except FileExistsError:
                # Already there — possibly a concurrent writer's. Not ours to
                # remove on rollback.
                continue
            self._created_dirs.append(d)

    def publish(self, staged: Path, target: Path) -> None:
        """Put ``staged`` at ``target`` (see the class docstring for how)."""
        if self.overwrite:
            ident = _identity(staged.stat())
            existed = target.exists()
            staged.replace(target)
            if not existed:
                self._created_files.append((target, ident))
            return

        if self._hardlinks_ok:
            ident = _identity(staged.stat())
            try:
                _hardlink(staged, target)
            except FileExistsError:
                raise _already_exists(target) from None
            except OSError as e:
                # EPERM / ENOTSUP / EXDEV ...: no hard links here. Use the
                # exclusive-create fallback for this and the remaining files.
                logger.debug(
                    "[SessionStore] export: hard link to %s refused (%s); "
                    "using exclusive create",
                    target,
                    e,
                )
                self._hardlinks_ok = False
            else:
                self._created_files.append((target, ident))
                with suppress(OSError):
                    staged.unlink()
                return

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(target, flags, 0o600)
        except FileExistsError:
            raise _already_exists(target) from None
        # Recorded before writing so a failed write is rolled back as well.
        self._created_files.append((target, _identity(os.fstat(fd))))
        with os.fdopen(fd, "wb") as f:
            f.write(staged.read_bytes())
        with suppress(OSError):
            target.chmod(0o600)
        with suppress(OSError):
            staged.unlink()

    def rollback(self) -> None:
        """Remove what :meth:`publish` / :meth:`ensure_dir` created, nothing else."""
        for path, ident in reversed(self._created_files):
            with suppress(OSError):
                if _identity(path.lstat()) == ident:
                    path.unlink()
        for d in reversed(self._created_dirs):
            with suppress(OSError):
                # Refuses a non-empty directory, so anyone else's files stay.
                d.rmdir()
