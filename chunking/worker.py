import os
import sys
from pathlib import Path
from typing import Union

from chunking.fallback import write_final_segs_ones


class _SilenceStdStreams:
    """
    Suppress both Python-level and OS-level stdout/stderr.
    This also silences output from native extensions (e.g., cut_pursuit_py).
    """

    def __enter__(self):
        import os as _os

        self._os = _os
        # Save Python-level streams
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        # Open devnull once
        self._devnull_file = open(_os.devnull, "w")
        # Redirect Python-level
        sys.stdout = self._devnull_file
        sys.stderr = self._devnull_file
        # Save OS-level fds
        self._old_stdout_fd = _os.dup(1)
        self._old_stderr_fd = _os.dup(2)
        # Redirect OS-level fds to devnull
        _os.dup2(self._devnull_file.fileno(), 1)
        _os.dup2(self._devnull_file.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore OS-level fds
        try:
            self._os.dup2(self._old_stdout_fd, 1)
        finally:
            try:
                self._os.dup2(self._old_stderr_fd, 2)
            finally:
                self._os.close(self._old_stdout_fd)
                self._os.close(self._old_stderr_fd)
        # Restore Python-level streams
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        # Close devnull
        self._devnull_file.close()
        return False


def run_treeiso_on_tile(tile_path: Union[str, Path]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Suppress at OS-level around both import and execution to silence native prints
    with _SilenceStdStreams():
        from PythonCpp.treeiso import process_las_file

    success = True
    try:
        with _SilenceStdStreams():
            process_las_file(str(tile_path))
    except Exception:
        success = False
    finally:
        if not success:
            write_final_segs_ones(tile_path)
    print(f"Finished processing tile {tile_path} ({'ok' if success else 'fallback'})")


