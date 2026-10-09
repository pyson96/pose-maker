#!/usr/bin/env python
"""Short load test of make_analysis.py with four camera inputs: does the PC keep up, and does
anything die?

    python load_test.py                 # 120 s, cameras.txt + each camera's stream7 -> 4 inputs
    python load_test.py --seconds 60 --variant stream7
    python load_test.py --inputs clip.mp4 clip.mp4 clip.mp4 clip.mp4   # files: as fast as the PC can go

The four inputs are the cameras.txt URLs plus, for each, the same URL with its stream path swapped
for --variant (e.g. .../stream2 -> .../stream7). make_analysis.py runs on them for --seconds while
GPU load, GPU memory, system RAM and the analysis processes are sampled every 2 s; the summary
says what peaked and whether every camera finished cleanly. Results go to a temp folder, never
into results/, and the test refuses to start while another analysis is running.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_analysis import default_inputs  # noqa: E402


def gpu_sample():
    """-> (util %, used MiB, total MiB) of GPU 0, or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, m, t = (float(v) for v in out.strip().splitlines()[0].split(","))
        return u, m, t
    except Exception:  # noqa: BLE001
        return None


def analysis_procs():
    """make_analysis.py processes and their children (workers, the shared-state manager) -- not every
    multiprocessing child on the machine (a training run elsewhere spawns those too)."""
    out = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        if any("make_analysis" in c for c in (p.info.get("cmdline") or [])):
            out[p.pid] = p
            try:
                for c in p.children(recursive=True):
                    out[c.pid] = c
            except psutil.Error:
                pass
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120, help="analysis time per run (model loading not included)")
    ap.add_argument("--variant", default="stream7", help="stream path added for each camera")
    ap.add_argument("--inputs", nargs="+", help="use these inputs instead (videos or streams, repeats allowed)")
    ap.add_argument("--model", default="yolo26l-pose.pt")
    ap.add_argument("--reid-interval", type=int, default=5, help="passed to make_analysis.py")
    ap.add_argument("--force", action="store_true", help="start even if another analysis is running")
    args = ap.parse_args()

    if args.inputs:
        inputs = args.inputs
    else:
        base = default_inputs()
        if not base:
            sys.exit("cameras.txt is missing or empty")
        extra = [re.sub(r"/stream\d+$", "/" + args.variant, u) if re.search(r"/stream\d+$", u)
                 else u.rstrip("/") + "/" + args.variant for u in base]
        inputs = base + extra
    if analysis_procs() and not args.force:
        sys.exit("another make_analysis.py is running -- not starting (use --force to override)")

    out_dir = os.path.join(tempfile.gettempdir(), "pm_load_test")
    mask = lambda s: re.sub(r"://([^:/@]+):[^@]+@", r"://\1:***@", s)
    print("inputs (%d):" % len(inputs))
    for i, u in enumerate(inputs, 1):
        print("  [%d] %s" % (i, mask(u)))
    print("output -> %s, %.0f s per camera, model %s, reid every %d frames\n"
          % (out_dir, args.seconds, args.model, args.reid_interval), flush=True)

    cmd = [sys.executable, os.path.join(HERE, "make_analysis.py"), "--input", *inputs,
           "--duration", str(args.seconds), "--out-dir", out_dir, "--model", args.model,
           "--reid-interval", str(args.reid_interval)]
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    lines = []

    def pump():
        for line in proc.stdout:
            lines.append(line.rstrip())
            if re.match(r"^\[\d\d:", line) or "Error" in line or "Traceback" in line:
                print("  | " + mask(line.rstrip()), flush=True)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    samples, peak_procs, worst_ram = [], 0, None
    total_ram = psutil.virtual_memory().total / 2**30
    while proc.poll() is None:
        g = gpu_sample()
        vm = psutil.virtual_memory()
        procs = analysis_procs()
        rss = 0.0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        samples.append({"t": time.time() - started, "gpu": g, "ram_free": vm.available / 2**30,
                        "ram_used_pct": vm.percent, "rss": rss / 2**30, "n": len(procs)})
        peak_procs = max(peak_procs, len(procs))
        worst_ram = vm.available / 2**30 if worst_ram is None else min(worst_ram, vm.available / 2**30)
        time.sleep(2)
    reader.join(timeout=5)
    wall = time.time() - started

    # --- summary
    busy = [s for s in samples if s["n"] >= 3]          # while the workers were up
    gpus = [s["gpu"] for s in busy if s["gpu"]]
    done = {}
    for l in lines:
        m = re.search(r"\[(\d)\] done: (\d+) frames read, (\d+) written, (\d+) people, ([\d.]+) s \(([\d.]+) fps\)", l)
        if m:
            done[int(m.group(1))] = (int(m.group(2)), float(m.group(5)), float(m.group(6)))
    failed = [l for l in lines if "cannot open" in l or "Traceback" in l or "Error" in l]
    print("\n================ summary ================")
    print("exit code        : %s   wall time %.0f s" % (proc.returncode, wall))
    for i, u in enumerate(inputs, 1):
        if i in done:
            n, secs, fps = done[i]
            print("camera %d          : OK   %d frames in %.0f s = %.1f fps   %s" % (i, n, secs, fps, mask(u)))
        else:
            print("camera %d          : NOT FINISHED (could not open, or died)   %s" % (i, mask(u)))
    if gpus:
        print("GPU utilisation  : avg %.0f %%, max %.0f %%" % (sum(g[0] for g in gpus) / len(gpus), max(g[0] for g in gpus)))
        print("GPU memory       : max %.1f / %.1f GB" % (max(g[1] for g in gpus) / 1024, gpus[0][2] / 1024))
    else:
        print("GPU              : no samples (nvidia-smi missing, or run too short)")
    if busy:
        print("analysis RAM     : max %.1f GB in %d processes (sum of RSS)" % (max(s["rss"] for s in busy), peak_procs))
    print("system RAM       : lowest free %.1f GB of %.1f GB (max used %.0f %%)"
          % (worst_ram or 0, total_ram, max((s["ram_used_pct"] for s in samples), default=0)))
    for l in failed[:10]:
        print("problem          : " + mask(l))
    ok = proc.returncode == 0 and len(done) == len(inputs)
    print("RESULT           : " + ("PASS -- every camera finished, nothing died" if ok else "CHECK -- see above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
