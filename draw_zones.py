#!/usr/bin/env python
"""Draw the CCTV 1 / CCTV 2 overlap zones on saved screenshots instead of the live cameras.

    python draw_zones.py                              # screen1.jpg + screen2.jpg -> overlap_zones.json
    python draw_zones.py --images a.jpg b.jpg --out zones.json

Screenshots are scaled to the analysis resolution (1280x720 by default) first, so the polygon points
match what make_analysis.py sees. Click to add points to the selected camera's polygon, r = redo it,
Enter = save, Esc = cancel. Copy the result next to make_analysis.py on the analysis PC.
"""

import argparse
import sys

import cv2

from make_analysis import draw_overlap_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs=2, default=["screen1.jpg", "screen2.jpg"], metavar=("CCTV1", "CCTV2"))
    p.add_argument("--size", default="1280x720", help="analysis resolution WxH the points are stored in")
    p.add_argument("--out", default="overlap_zones.json")
    args = p.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    frames = []
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            p.error("cannot read " + path)
        frames.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC))
    if draw_overlap_config(frames, args.out) is None:
        print("cancelled -- nothing saved")
        sys.exit(1)


if __name__ == "__main__":
    main()
