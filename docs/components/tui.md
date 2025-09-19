# TUI

- Purpose: Textual-based terminal UI to run and monitor missions with interactive panels and stats.
- Key interfaces: `seek.components.tui.dataseek_tui.DataSeekTUI` with supporting components and managers.
- Dependencies: `textual`, `typer`, `seek.common.config` for active configuration, `seek.components.mission_runner` for state references.
- Configuration: Uses `seek.common.config.load_seek_config`/`set_active_seek_config` and mission plan paths.
- Minimal usage:
  - `python -m seek.components.tui.dataseek_tui` to launch, or import `DataSeekTUI` and call `.run()`.
