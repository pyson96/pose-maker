#!/usr/bin/env python
"""Mark the entrance / exit floor and wall on CCTV screenshots, for calibrating a camera against them.

    python draw_marks.py                                   # screen_101_stream7.jpg, screen_103_stream7.jpg
    python draw_marks.py --images a.jpg b.jpg --out cam_marks.json

Each image is shown at the analysis resolution (1280x720 by default; points are stored in it).
Each image shows one doorway: --gates exit entrance (101 stream7 sees the exit, 103 stream7 the entrance).
Keys: 1 floor  2 wall   (of that doorway; selects what the clicks draw)
      click = add point   Backspace = undo point   r = clear the selected mark
      Enter = next image (saves after the last)   Esc = cancel without saving
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

MARKS = [  # key, name, BGR colour
    ("1", "entrance_floor", (110, 210, 60)),
    ("2", "entrance_wall", (40, 120, 20)),
    ("3", "exit_floor", (40, 140, 245)),
    ("4", "exit_wall", (20, 70, 180)),
]


def mark_image(img, title, gate):
    """-> {mark name: [[x, y], ...]} for this image's doorway (gate_floor / gate_wall), or None if cancelled."""
    kinds = [(str(i + 1), name, col) for i, (_, name, col) in enumerate(m for m in MARKS if m[1].startswith(gate))]
    marks = {name: [] for _, name, _ in kinds}
    state = {"cur": kinds[0][1]}
    scale = min(1.0, 1200.0 / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            marks[state["cur"]].append([int(x / scale), int(y / scale)])

    win = "marks: " + title
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            canvas = view.copy()
            for _, name, col in kinds:
                pts = (np.array(marks[name], np.float32).reshape(-1, 2) * scale).astype(np.int32)
                if len(pts) >= 3:
                    fill = canvas.copy()
                    cv2.fillPoly(fill, [pts], col)
                    canvas = cv2.addWeighted(fill, 0.3, canvas, 0.7, 0)
                if len(pts):
                    cv2.polylines(canvas, [pts], len(pts) >= 3, col, 2, cv2.LINE_AA)
                    for q in pts:
                        cv2.circle(canvas, (int(q[0]), int(q[1])), 4, col, -1, cv2.LINE_AA)
            y = 22
            for key, name, col in kinds:
                sel = name == state["cur"]
                txt = "%s %s (%d pts)%s" % (key, name.replace("_", " "), len(marks[name]), "  <" if sel else "")
                cv2.rectangle(canvas, (6, y - 16), (16 + 9 * len(txt), y + 6), (0, 0, 0), -1)
                cv2.putText(canvas, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col if not sel else (255, 255, 255), 1, cv2.LINE_AA)
                y += 24
            cv2.putText(canvas, "click: point | Backspace: undo | r: clear | Enter: done | Esc: cancel",
                        (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            k = cv2.waitKey(30) & 0xFF
            if k == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            for key, name, _ in kinds:
                if k == ord(key):
                    state["cur"] = name
            if k == 8 and marks[state["cur"]]:
                marks[state["cur"]].pop()
            if k in (ord("r"), ord("R")):
                marks[state["cur"]] = []
            if k in (13, 10):
                return {n: p for n, p in marks.items() if len(p) >= 3}
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:                 # already closed with the window's X
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["screen_101_stream7.jpg", "screen_103_stream7.jpg"])
    ap.add_argument("--gates", nargs="+", default=["exit", "entrance"], choices=["exit", "entrance"],
                    help="which doorway each image shows")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--out", default="cam_marks.json")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    out = {}
    for i, path in enumerate(a.images):
        img = cv2.imread(path)
        if img is None:
            sys.exit("cannot read " + path)
        res = mark_image(cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC), os.path.basename(path),
                         a.gates[min(i, len(a.gates) - 1)])
        if res is None:
            print("cancelled -- nothing saved")
            sys.exit(1)
        out[os.path.basename(path)] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"size": [w, h], "images": out}, f, indent=1)
    print("saved -> " + a.out)
    for k, v in out.items():
        print("  %s: %s" % (k, ", ".join("%s %d pts" % (n, len(p)) for n, p in v.items()) or "nothing"))


if __name__ == "__main__":
    main()
