"""Re-simulates the inputs of the last replayed job (replay/job.done.txt) with the current simulator.

The game's traces in replay/out/ stay valid for those inputs, so after a simulator change this rewrites
replay/sim/<name>.csv and scripts/compare_trace.py shows whether the simulator now follows the game longer,
without replaying anything in the game.

Usage: python scripts/resim_job.py && python scripts/compare_trace.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_replay_job import REPLAY_DIR, TRACE_HEADER, sim_trace  # noqa: E402

from gdgen.levelstring import decode_level_string, parse_level  # noqa: E402
from gdgen.sim.collide import Simulator  # noqa: E402
from gdgen.sim.level import build_level  # noqa: E402


def holds_from_events(events: list[str], total: int) -> list[bool]:
    changes = sorted((int(t), v == "1") for t, v in (e.split(":") for e in events))
    holds, cur, i = [], False, 0
    for tick in range(total):
        while i < len(changes) and changes[i][0] <= tick:
            cur = changes[i][1]
            i += 1
        holds.append(cur)
    return holds


def main() -> int:
    lines = (REPLAY_DIR / "job.done.txt").read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("job "):
            continue
        _, name, level_file, _song, _track, max_ticks, _trace = line.split()
        events = lines[i + 1].split()[1:] if i + 1 < len(lines) and lines[i + 1].startswith("inputs") else []
        header, objects = parse_level(decode_level_string((REPLAY_DIR / level_file).read_text()))
        sim = Simulator(build_level(header, objects))
        holds = holds_from_events(events, int(max_ticks))
        (REPLAY_DIR / "sim" / f"{name}.csv").write_text(TRACE_HEADER + "\n" + "\n".join(sim_trace(sim, holds)) + "\n")
        print(f"{name}: re-simulated {len(holds)} ticks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
