# Vector Racer

Top-down arcade-style racing game written in Python with pygame. Drive through three themed circuits featuring differing grip levels, track geometries, and checkpointed lap timing.

## Requirements

- Python 3.9+
- pygame (install with `pip install pygame`)

## Running

```bash
python main.py
```

Controls use the arrow keys or WASD to steer, `Up`/`W` to accelerate, `Down`/`S` to reverse, `Shift` for heavy braking, and `Space` for a quick handbrake flick. Press `1`, `2`, or `3` in the menu to jump directly to a track, and `Esc`/`P` pauses the action in-race.

For quick iteration on the driving feel without launching pygame, run the deterministic control simulations:

```bash
python simulate_controls.py
```

The script exercises straight-line acceleration, sweeping turns, braking, and handbrake maneuvers while printing the resulting speeds and headings.

Lap records are saved automatically in `records.json` next to the script.
