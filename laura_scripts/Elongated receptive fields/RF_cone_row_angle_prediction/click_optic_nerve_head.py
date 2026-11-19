"""
Click the optic nerve head (ONH) on a recording's cone image.

Writes the click to optic_nerve_head_positions.json (next to this script), in pixels of
the door-view (flipped) cone image that the alignment uses
(alignment_recordings.cone_image_file, e.g. 14_08_2026_p0_cones_flipped.jpg).
rf_cone_row_angle.py reads it from there.

Run from the repo root, one recording at a time:
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/click_optic_nerve_head.py" --recording 14_08_2026

In the window:
  * zoom / pan with the toolbar as you like (switch the zoom tool OFF again before clicking)
  * LEFT click on the ONH centre (click again to move it; RIGHT click removes the last click)
  * press ENTER to accept the last click
  * a second window shows the point: press Y to save it, N (or close) to discard
The image is shown with a grey margin so you can click outside it if the ONH is just off
the image. If you'd rather type the coordinates, edit the JSON by hand:
    "14_08_2026": {"x_px": 5230.0, "y_px": 812.0, "note": "..."}
"""

import argparse
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from laura_scripts.Alignment_pipeline.alignment_recordings import cone_image_file  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

ONH_FILE = Path(__file__).resolve().parent / "optic_nerve_head_positions.json"
DISPLAY_DOWNSAMPLE = 4  # the full images are ~11000 px square; clicks are still in full-size px
MARGIN_FRACTION = 0.25  # grey margin around the image, so the ONH can be clicked just off it


def load_onh_positions() -> dict:
    if ONH_FILE.exists():
        return json.loads(ONH_FILE.read_text())
    return {}


parser = argparse.ArgumentParser()
parser.add_argument("--recording", required=True, help="DD_MM_YYYY, e.g. 14_08_2026")
parser.add_argument("--note", default="", help="optional note stored with the click")
args = parser.parse_args()

image_path = cone_image_file(args.recording)
print(f"Cone image: {image_path}")
img = Image.open(image_path)
full_w, full_h = img.size
small = np.asarray(img.convert("RGB").reduce(DISPLAY_DOWNSAMPLE))

positions = load_onh_positions()
previous = positions.get(args.recording)

fig, ax = plt.subplots(figsize=(11, 11))
ax.set_facecolor("0.6")
# extent in FULL-SIZE pixel coordinates, so ginput returns full-size pixels.
# ORIENTATION [DISPLAY only]: origin="upper", y down, exactly as the image file.
ax.imshow(small, extent=[0, full_w, full_h, 0], origin="upper", interpolation="nearest")
mx, my = MARGIN_FRACTION * full_w, MARGIN_FRACTION * full_h
ax.set_xlim(-mx, full_w + mx)
ax.set_ylim(full_h + my, -my)
if previous is not None:
    ax.plot(previous["x_px"], previous["y_px"], "c+", ms=25, mew=2, label="previous click")
    ax.legend(loc="upper right")
ax.set_title(f"{args.recording}: LEFT click the optic nerve head, then ENTER\n"
             "(right click = undo; turn zoom/pan off before clicking)")
clicks = plt.ginput(n=-1, timeout=0, show_clicks=True)
plt.close(fig)

if not clicks:
    raise SystemExit("No click; nothing saved.")
x_px, y_px = (float(v) for v in clicks[-1])
inside = 0 <= x_px < full_w and 0 <= y_px < full_h
print(f"ONH at x = {x_px:.1f}, y = {y_px:.1f} cone-image px ({'inside' if inside else 'OUTSIDE'} the image)")

# Show the result before saving
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_facecolor("0.6")
ax.imshow(small, extent=[0, full_w, full_h, 0], origin="upper")
ax.plot(x_px, y_px, "r+", ms=30, mew=2.5)
ax.set_xlim(-mx, full_w + mx)
ax.set_ylim(full_h + my, -my)
ax.set_title(f"{args.recording}: ONH (red +).\nPress Y to SAVE, N (or close the window) to discard")
answer = {"save": False}


def on_key(event):
    if event.key in ("y", "Y"):
        answer["save"] = True
        plt.close(fig)
    elif event.key in ("n", "N"):
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()

if not answer["save"]:
    raise SystemExit("Not saved.")

positions = load_onh_positions()  # re-read in case it changed meanwhile
positions[args.recording] = dict(
    x_px=round(x_px, 1),
    y_px=round(y_px, 1),
    image=image_path.name,
    image_size_px=[full_w, full_h],
    clicked=datetime.date.today().isoformat(),
    note=args.note,
)
ONH_FILE.write_text(json.dumps(dict(sorted(positions.items())), indent=2) + "\n")
print(f"Saved to {ONH_FILE}")
