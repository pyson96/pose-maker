#!/usr/bin/env python
"""Record the cameras' original streams to disk, without re-encoding (PyAV remux: a few % of one CPU core).

    python record.py --until 19:00                   # cameras.txt -> recordings/YYYY-MM-DD/N_YYMMDD_HHMMSS.mp4
    python record.py --input rtsp://... --duration 60

One thread per camera. Files are fragmented MP4 (video only), cut every --segment seconds at a keyframe,
so a crash or power cut loses at most the fragment being written and every file plays in a browser.
Day folders older than --keep-days are deleted at start and at each cut, and while the disk has less
than --min-free-gb free the oldest day folders go first (never today's).
"""

import argparse
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta

import av

from make_analysis import default_inputs, log

RECONNECT_SEC = 5.0
OPEN_OPTIONS = {"rtsp_transport": "tcp", "timeout": "10000000"}   # socket timeout in microseconds
MP4_OPTIONS = {"movflags": "frag_keyframe+empty_moov+default_base_moof"}


def safe(url):
    """URL for the log, without the camera password."""
    return re.sub(r"//[^/@]*@", "//***@", url)


def day_dirs(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))


_warned = {"low_disk": False}


def clean_up(root, keep_days, min_free_gb, lock=threading.Lock()):
    """Drop day folders older than keep_days, then the oldest ones while the disk is short of space."""
    with lock:
        today = datetime.now().strftime("%Y-%m-%d")
        oldest_kept = (datetime.now() - timedelta(days=keep_days - 1)).strftime("%Y-%m-%d")
        for d in day_dirs(root):
            if d < oldest_kept:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                log("[rec] removed %s (older than %d days)" % (d, keep_days))
        while shutil.disk_usage(root).free < min_free_gb * 1024 ** 3:
            old = [d for d in day_dirs(root) if d != today]
            if not old:
                if not _warned["low_disk"]:
                    log("[rec] disk below %.0f GB free and nothing old to remove" % min_free_gb)
                    _warned["low_disk"] = True
                break
            shutil.rmtree(os.path.join(root, old[0]), ignore_errors=True)
            log("[rec] removed %s (disk below %.0f GB free)" % (old[0], min_free_gb))


def record_camera(idx, url, args, deadline):
    tag = "[rec %d] " % idx
    while time.time() < deadline:
        try:
            src = av.open(url, options=OPEN_OPTIONS)
        except Exception as e:                      # camera off, network down, too many clients ...
            log(tag + "cannot open %s: %s -- retry in %.0f s" % (safe(url), e, RECONNECT_SEC))
            time.sleep(RECONNECT_SEC)
            continue
        vin = src.streams.video[0]
        log(tag + "%s: %s %dx%d" % (safe(url), vin.codec_context.name, vin.codec_context.width, vin.codec_context.height))
        out = vout = None
        base = None
        started = 0.0
        try:
            for pkt in src.demux(vin):
                if pkt.dts is None or pkt.size == 0:
                    continue
                now = time.time()
                if now >= deadline:
                    break
                # a new file at the first keyframe after the segment length (or the very first keyframe)
                if pkt.is_keyframe and (out is None or now - started >= args.segment):
                    if out is not None:
                        out.close()
                    clean_up(args.out_dir, args.keep_days, args.min_free_gb)
                    stamp = datetime.now()
                    folder = os.path.join(args.out_dir, stamp.strftime("%Y-%m-%d"))
                    os.makedirs(folder, exist_ok=True)
                    name = "%d_%s" % (idx, stamp.strftime("%y%m%d_%H%M%S"))
                    path, k = os.path.join(folder, name + ".mp4"), 1
                    while os.path.exists(path):          # two cuts in one second (reconnects, tests)
                        k += 1
                        path = os.path.join(folder, "%s_%d.mp4" % (name, k))
                    out = av.open(path, "w", format="mp4", options=MP4_OPTIONS)
                    vout = out.add_stream_from_template(vin)
                    base, started = pkt.dts, now
                if out is None:                          # wait for the first keyframe
                    continue
                pkt.dts -= base
                pkt.pts = pkt.dts if pkt.pts is None else pkt.pts - base
                if pkt.dts < 0:
                    continue
                pkt.stream = vout
                out.mux(pkt)
            else:
                log(tag + "stream ended -- reconnecting")
        except Exception as e:
            log(tag + "stream lost (%s) -- reconnecting" % e)
        finally:
            if out is not None:
                out.close()
            src.close()
        if time.time() < deadline:
            time.sleep(RECONNECT_SEC)
    log(tag + "stop time reached")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", default=default_inputs(), help="default: cameras.txt")
    p.add_argument("--out-dir", default="recordings")
    p.add_argument("--until", help="stop at this wall-clock time, HH:MM")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = --until or forever)")
    p.add_argument("--segment", type=float, default=600, help="seconds per file (cut at the next keyframe)")
    p.add_argument("--keep-days", type=int, default=14, help="day folders kept, today included")
    p.add_argument("--min-free-gb", type=float, default=30, help="delete the oldest days below this free space")
    args = p.parse_args()
    if not args.input:
        p.error("no --input given and cameras.txt is missing or empty")
    deadline = float("inf")
    if args.duration:
        deadline = time.time() + args.duration
    if args.until:
        h, m = (int(v) for v in args.until.split(":"))
        stop = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= datetime.now():
            sys.exit("--until %s is already past" % args.until)
        deadline = min(deadline, stop.timestamp())
    os.makedirs(args.out_dir, exist_ok=True)
    clean_up(args.out_dir, args.keep_days, args.min_free_gb)
    log("[rec] %d camera(s) -> %s, %.0f s files, keep %d days" % (len(args.input), os.path.abspath(args.out_dir), args.segment, args.keep_days))
    threads = [threading.Thread(target=record_camera, args=(i, url, args, deadline), daemon=True)
               for i, url in enumerate(args.input, 1)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log("[rec] interrupted")


if __name__ == "__main__":
    main()
