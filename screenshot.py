#!/usr/bin/env python
"""Grab the current frame of each camera -> screen1.jpg, screen2.jpg in the current folder.

    python screenshot.py                 # cameras.txt
    python screenshot.py --input a b     # other streams/videos
"""

import argparse
import os
import sys

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2

from make_analysis import default_inputs


def grab(path, skip):
    """Return one current frame, or None. A few frames are discarded so the decoder settles on a keyframe."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(skip + 1):
            ok, f = cap.read()
            if ok:
                frame = f
        return frame
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", default=default_inputs(), help="default: cameras.txt")
    p.add_argument("--skip", type=int, default=5, help="frames to discard before saving")
    args = p.parse_args()
    if not args.input:
        p.error("no --input given and cameras.txt is missing or empty")

    failed = 0
    for i, path in enumerate(args.input, 1):
        out = "screen%d.jpg" % i
        frame = grab(path, args.skip)
        if frame is None:
            print("[%d] failed: %s" % (i, path))
            failed += 1
            continue
        cv2.imwrite(out, frame)
        print("[%d] %dx%d -> %s" % (i, frame.shape[1], frame.shape[0], os.path.abspath(out)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
