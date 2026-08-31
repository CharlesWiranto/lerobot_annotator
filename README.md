# LeRobot annotator

Local web app for labeling `instruction_segments` (and optional `high_level_instruction`) on LeRobot v2.1 datasets in the parent folder.

## Requirements

- Python 3.10+
- The packages in `requirements.txt` (`fastapi`, `uvicorn`, `opencv-python-headless`, `numpy`)

From this directory:

```powershell
pip install -r requirements.txt
```

## Run

**Windows:** double-click `run.bat`, or:

```powershell
cd D:\1Research\1Robot\dataset\annotator
python app.py
```

Then open **http://127.0.0.1:8765**

The server uses the **parent of this folder** as the dataset root (`D:\1Research\1Robot\dataset`). Any sibling directory that contains `meta/info.json` shows up on the picker (e.g. `pick_give_beverage_sim_part1`, `pick_give_beverage_sim_part2`).

Stop the server with `Ctrl+C` in the terminal.

## Usage (short)

1. Choose a dataset.
2. Set the **task template**: number of segments `N` and the default instruction for each.
3. **Apply to this episode** or **Apply to unreviewed** (even split: first `start_frame_index = 0`, last `end_frame_index = length` from `episodes.jsonl`).
4. Scrub frames (all cameras stay in sync). Mark **start / success / end** on the active segment (`S` / `X` / `E`), or drag the timeline handles.
5. Edit per-segment text if this episode differs from the template.
6. **Save to info.json** (`Ctrl+S`).

`end_frame_index` is exclusive (last included frame is `end - 1`). Training samples `[start, success]`.

If `high_level_instruction` is non-empty, training **replaces** every subtask string with that HL text. Use **Clear HL on all episodes** (and template HL mode **Off**) if you want the model to see per-segment instructions.

## What gets written

| File | Role |
|---|---|
| `<dataset>/meta/info.json` | Updates `instruction_segments` and `high_level_instruction` only |
| `<dataset>/meta/info.json.bak` | Copy of `info.json` from just before the last save |
| `<dataset>/meta/annotation_progress.json` | Reviewed flags and the task template (not used by training) |
| `<dataset>/broken.json` | Episode indices to skip; toggle **Broken** or press `B` (written immediately) |

## Keys

| Key | Action |
|---|---|
| `←` `→` | ±1 frame |
| `Shift` + arrows | ±N frames (jump size in the toolbar) |
| `Space` | play / pause at dataset FPS (30) |
| `[` `]` | previous / next episode |
| `1`–`9` | select segment |
| `S` `X` `E` | set start / success / end of the active segment at the current frame |
| `Home` `End` | first / last frame |
| `Ctrl+S` | save |
