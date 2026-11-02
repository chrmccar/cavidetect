"""Single entry point for the packaged (PyInstaller) application.

A frozen build is one executable, but the GUI needs to spawn the pipeline as a
child process (so Cancel can terminate it) and a frozen app has no `python -m`.
So this one executable is dual-purpose: if its first argument is a known CLI
subcommand it runs the pipeline; otherwise it launches the GUI. `gui.build_command`
produces `<exe> colour-shape …` when frozen, which lands here in worker mode.
"""
from __future__ import annotations
import sys

_CLI_CMDS = {"run", "batch", "colour-shape"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in _CLI_CMDS:
        from .cli import main as cli_main
        cli_main(argv)
        return
    from .gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
