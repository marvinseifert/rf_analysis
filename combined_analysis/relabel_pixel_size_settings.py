"""
Relabel a recording's stimulus pixel size in its settings files (e.g. 2.5 -> 2.0 µm).

Rule (Laura, 18/11/2026): recordings from 20/04/2026 use 2.0 µm stimulus pixels.
Some were analysed with 2.5 µm; this fixes their settings so the v2 pipeline,
noise_data_v2.nc (convert_noise_data_v2.py --pixel-size 2.0) and the alignment
all agree. The old noise_data.nc is not touched.

For one recording (all Phase_* folders):
  * <stimulus>_idx_<n>/noise_params.json      noise_pixel_size  OLD -> NEW
  * noise_analysis*/recording_config.json      every "pixel_size" OLD -> NEW
  * noise_analysis*/collapse_2d_config.json    cut_size_um x NEW/OLD (cut in pixels unchanged)
Each file is backed up first as <name>_orig_<OLD>um.json. A file whose backup
already exists is skipped, so running it twice does nothing. Only values equal
to OLD are changed.

    poetry run python -m combined_analysis.relabel_pixel_size_settings zebrafish_21_04_2026 --old 2.5 --new 2.0
"""
import argparse
import json
import shutil
from pathlib import Path

NAS_LAURA = Path("/mnt/mea_nas/Laura")


def _backup_path(f: Path, old: float) -> Path:
    return f.with_name(f"{f.stem}_orig_{old:g}um{f.suffix}")


def _replace_key(obj, key, old, new):
    """Recursively set obj[...][key] = new where it equals old. Returns the count."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)) and abs(v - old) < 1e-9:
                obj[k] = new
                n += 1
            else:
                n += _replace_key(v, key, old, new)
    elif isinstance(obj, list):
        for v in obj:
            n += _replace_key(v, key, old, new)
    return n


def _edit(f: Path, old: float, edit, dry_run: bool) -> str:
    backup = _backup_path(f, old)
    if backup.exists():
        return f"SKIP (backup exists)  {f}"
    data = json.loads(f.read_text())
    n = edit(data)
    if n == 0:
        return f"no change             {f}"
    if not dry_run:
        shutil.copy2(f, backup)
        f.write_text(json.dumps(data, indent=4))
    return f"{'would edit' if dry_run else 'EDITED'} ({n})          {f}"


def relabel(recording: str, old: float, new: float, dry_run: bool = False) -> None:
    rec = NAS_LAURA / recording
    if not rec.is_dir():
        raise FileNotFoundError(rec)
    scale = new / old

    def noise_params(d):
        return _replace_key(d, "noise_pixel_size", old, new)

    def recording_config(d):
        return _replace_key(d, "pixel_size", old, new)

    def collapse(d):
        cut = d.get("cut_size_um")
        if not isinstance(cut, dict) or "data" not in cut:
            return 0
        cut["data"] = [round(v * scale, 6) for v in cut["data"]]
        return len(cut["data"])

    for f in sorted(rec.glob("Phase_*/*_idx_*/noise_params.json")):
        print(_edit(f, old, noise_params, dry_run))
    for folder in sorted(rec.glob("Phase_*/noise_analysis*/")):
        if not (folder / "noise_data.nc").exists():
            continue
        rc = folder / "recording_config.json"
        if rc.exists():
            print(_edit(rc, old, recording_config, dry_run))
        # Only rescale the cut if this folder's recording_config used OLD (backup now exists).
        cc = folder / "collapse_2d_config.json"
        if cc.exists() and _backup_path(rc, old).exists():
            print(_edit(cc, old, collapse, dry_run))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("recording", help="e.g. zebrafish_21_04_2026")
    p.add_argument("--old", type=float, default=2.5)
    p.add_argument("--new", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    relabel(a.recording, a.old, a.new, a.dry_run)
