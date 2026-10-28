#!/usr/bin/env python
"""Pick named calibration points on CCTV screenshots, one after another, with on-screen prompts.

    python draw_points.py                                  # screen_101_stream7.jpg, screen_103_stream7.jpg
    python draw_points.py --images a.jpg --out cam_points.json

For each image (the side wall next to the doorway it shows):
    1 corner_floor  where the wall ends at the doorway, on the floor
    2 corner_top    the same vertical edge, as high as it is visible
    3 line_far      on the wall/floor line, far from the corner
    4 line_mid      on the wall/floor line, between 1 and 3
    5 extra         (optional, s = skip) any other point whose floor-plan position you know
Points are stored at the analysis resolution (1280x720 by default).
Click = place the prompted point, Backspace = undo, s = skip the optional one,
Enter = next image (needs points 1-4), Esc / closing the window = cancel.
"""

import argparse
import json
import os
import sys

import cv2

STEPS = [
    ("corner_floor", "1  wall END at the doorway -- on the FLOOR", (60, 220, 60)),
    ("corner_top", "2  same vertical edge -- as HIGH as visible", (40, 160, 255)),
    ("line_far", "3  wall/floor line -- FAR from the corner", (255, 120, 40)),
    ("line_mid", "4  wall/floor line -- between 1 and 3", (230, 60, 200)),
    ("extra", "5  (optional, s = skip) a point whose plan position you know", (0, 220, 255)),
]
REQUIRED = 4


def pick(img, title, hint):
    pts = {}
    scale = min(1.0, 1200.0 / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale)
    state = {"i": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(STEPS):
            pts[STEPS[state["i"]][0]] = [int(x / scale), int(y / scale)]
            state["i"] += 1

    win = "points: " + title
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            canvas = view.copy()
            for name, label, col in STEPS:
                if name in pts:
                    p = (int(pts[name][0] * scale), int(pts[name][1] * scale))
                    cv2.circle(canvas, p, 7, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, p, 5, col, -1, cv2.LINE_AA)
                    cv2.putText(canvas, label.split()[0], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            if "corner_floor" in pts and "corner_top" in pts:
                a, b = (tuple(int(v * scale) for v in pts[k]) for k in ("corner_floor", "corner_top"))
                cv2.line(canvas, a, b, (40, 160, 255), 1, cv2.LINE_AA)
            line = [tuple(int(v * scale) for v in pts[k]) for k in ("corner_floor", "line_mid", "line_far") if k in pts]
            for a, b in zip(line, line[1:]):
                cv2.line(canvas, a, b, (255, 120, 40), 1, cv2.LINE_AA)
            i = state["i"]
            msg = ("NOW: " + STEPS[i][1]) if i < len(STEPS) else "all points placed -- Enter to continue"
            for k, txt in enumerate((hint, msg, "Backspace undo | s skip optional | Enter next | Esc cancel")):
                y = 24 + 26 * k
                cv2.rectangle(canvas, (6, y - 18), (14 + 10 * len(txt), y + 7), (0, 0, 0), -1)
                cv2.putText(canvas, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            STEPS[i][2] if (k == 1 and i < len(STEPS)) else (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == 8 and state["i"] > 0:
                state["i"] -= 1
                pts.pop(STEPS[state["i"]][0], None)
            if key in (ord("s"), ord("S")) and state["i"] == REQUIRED:
                state["i"] += 1
            if key in (13, 10) and state["i"] >= REQUIRED:
                return pts
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["screen_101_stream7.jpg", "screen_103_stream7.jpg"])
    ap.add_argument("--hints", nargs="+", default=["CCTV 3 (101 stream7): EXIT side wall", "CCTV 4 (103 stream7): ENTRANCE side wall"])
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--out", default="cam_points.json")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    out = {}
    for i, path in enumerate(a.images):
        img = cv2.imread(path)
        if img is None:
            sys.exit("cannot read " + path)
        res = pick(cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC), os.path.basename(path),
                   a.hints[i] if i < len(a.hints) else os.path.basename(path))
        if res is None:
            print("cancelled -- nothing saved")
            sys.exit(1)
        out[os.path.basename(path)] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"size": [w, h], "images": out}, f, indent=1)
    print("saved -> " + a.out)
    for k, v in out.items():
        print("  %s: %s" % (k, ", ".join("%s %s" % (n, p) for n, p in v.items())))


if __name__ == "__main__":
    main()
