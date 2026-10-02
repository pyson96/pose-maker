#!/usr/bin/env python
"""Minimal video -> gzipped analysis JSON (YOLO26-Pose + ByteTrack + SOLIDER ReID), for bakery.html.

    python make_analysis.py --input a.mp4 b.mp4
        -> 1_YYMMDD.json.gz, 2_YYMMDD.json.gz   (inputs processed concurrently)
    python make_analysis.py --input rtsp://... --save-video
        -> 1_YYMMDD.json.gz + 1_YYMMDD.mp4      (Ctrl+C to stop; both files are closed cleanly)
    python make_analysis.py
        -> analyses the cameras listed in cameras.txt

Keypoints are a flat array in COCO-17 order: [x0, y0, conf0, x1, y1, conf1, ...].
track_id is ByteTrack's temporary id; global_id is a ReID identity that survives exits and re-entries
(null until the track's first ReID frame).
Frames are streamed to disk as they are produced, so memory stays flat.
"""

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime

import cv2
import numpy as np

KPT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
REID_SIZE = (128, 384)       # SOLIDER input (w, h)
REID_SEMANTIC = 0.2          # SOLIDER semantic weight used for its released ReID models
MIN_CROP_W, MIN_CROP_H = 16, 32
CAMERAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.txt")


def default_inputs():
    """Default inputs from cameras.txt (one URL per line, # comments); kept out of git since URLs hold passwords."""
    if not os.path.isfile(CAMERAS_FILE):
        return []
    with open(CAMERAS_FILE, encoding="utf-8") as f:
        return [s for s in (line.strip() for line in f) if s and not s.startswith("#")]

COORDINATE_SYSTEM = {
    "image_origin": "top-left",
    "x_direction": "right",
    "y_direction": "down",
    "bbox_unit": "pixel",
    "keypoint_unit": "pixel",
    "ground_point_unit": "pixel",
    "head_direction": "normalized_image_vector",
    "keypoint_format": "flat_xyc",
    "keypoint_names": list(KPT_NAMES),
}


def head_direction(kp, conf, thr, fov):
    """Estimated head direction from nose vs. ears/eyes/shoulders axis."""
    if conf[NOSE] < thr:
        return None
    nose = kp[NOSE].astype(np.float64)
    for source, li, ri in (("ears", L_EAR, R_EAR), ("eyes", L_EYE, R_EYE), ("shoulders", L_SH, R_SH)):
        if conf[li] < thr or conf[ri] < thr:
            continue
        left, right = kp[li].astype(np.float64), kp[ri].astype(np.float64)
        axis = right - left
        axis_len = float(np.hypot(*axis))
        if axis_len < 1e-3:
            continue
        axis /= axis_len
        middle = (left + right) / 2.0
        offset = nose - middle
        u = max(-1.0, min(1.0, float(offset @ axis) / (axis_len / 2.0)))
        perp = np.array([axis[1], -axis[0]])
        side = float(offset @ perp)
        if source == "shoulders" or abs(side) < 1e-6:
            if perp[1] < 0:
                perp = -perp
        elif side < 0:
            perp = -perp
        vec = u * axis + math.sqrt(max(0.0, 1.0 - u * u)) * perp
        norm = float(np.hypot(*vec))
        if norm < 1e-6:
            continue
        vec /= norm
        origin = nose if source == "shoulders" else middle
        return {
            "origin": [round(float(origin[0]), 3), round(float(origin[1]), 3)],
            "direction": [round(float(vec[0]), 6), round(float(vec[1]), 6)],
            "angle_deg": round(math.degrees(math.atan2(-vec[1], vec[0])), 3),
            "source": source,
            "fov_deg": fov,
        }
    return None


def load_reid(args):
    """SOLIDER Swin-Small backbone with fine-tuned ReID weights, in eval mode on the YOLO device."""
    import importlib.util
    import sys
    import types
    import torch

    # The backbone imports mmcv only for training-time checkpoint loading, which we never call.
    if importlib.util.find_spec("mmcv") is None:
        sys.modules["mmcv"] = types.ModuleType("mmcv")
        sys.modules["mmcv.runner"] = types.SimpleNamespace(load_checkpoint=None)
    spec = importlib.util.spec_from_file_location(
        "solider_swin", os.path.join(args.reid_repo, "model", "backbones", "swin_transformer.py"))
    swin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swin)

    model = swin.swin_small_patch4_window7_224(img_size=REID_SIZE[::-1], semantic_weight=REID_SEMANTIC)
    sd = torch.load(args.reid_weights, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1)[len("base."):]: v for k, v in sd.get("state_dict", sd).items()
          if k.replace("module.", "", 1).startswith("base.")}  # backbone only; drop BNNeck/classifier
    model.load_state_dict(sd)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:" + args.device)
    model.to(device).eval()  # SOLIDER's train() override returns None, so don't chain
    return model, device


def reid_embed(reid, crops):
    """One batched forward over all crops -> L2-normalised embeddings (N, 768)."""
    import torch

    model, device = reid
    x = torch.from_numpy(np.stack([cv2.resize(c, REID_SIZE)[:, :, ::-1] for c in crops]))
    x = x.to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0  # mean = std = 0.5
    w = torch.tensor([[REID_SEMANTIC, 1.0 - REID_SEMANTIC]], device=device).expand(len(crops), 2)
    # FP16 via autocast: Swin's attention mixes in fp32 tensors, so a plain .half() model fails.
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        feat = model(x, w)[0]
    return torch.nn.functional.normalize(feat.float(), dim=1).cpu().numpy()


def l2(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def persons_of(result, args, track_state):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else None
    confs = boxes.conf.cpu().numpy()
    kxy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    kcf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else None

    out = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy[:, :4].astype(float)):
        keypoints, head = [], None
        if kxy is not None:
            kc = kcf[i] if kcf is not None else np.ones(len(kxy[i]))
            for j in range(len(KPT_NAMES)):
                keypoints += [round(float(kxy[i][j][0]), 3), round(float(kxy[i][j][1]), 3), round(float(kc[j]), 4)]
            head = head_direction(kxy[i], kc, args.kpt_conf, args.view_fov)
        tid = int(ids[i]) if ids is not None else -1
        out.append({
            "track_id": tid,
            "global_id": track_state[tid]["global_id"] if tid in track_state else None,
            "bbox": {"x1": round(x1, 3), "y1": round(y1, 3), "x2": round(x2, 3), "y2": round(y2, 3)},
            "det_conf": round(float(confs[i]), 4),
            "ground_point": [round((x1 + x2) / 2.0, 3), round(y2, 3)],
            "keypoints": keypoints,
            "head": head,
        })
    return out


def run(order, path, output, args):
    """Analyse one video and stream its frames into `output` (gzip)."""
    from ultralytics import YOLO

    tag = "[%d] " % order
    if path.lower().startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(tag + "cannot open video: " + path)
        return
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:  # live streams often report 0 or a 90 kHz clock
        fps = 30.0
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # 0 for live streams
    print(tag + "%s: %dx%d @ %.3f fps, %d frames -> %s" % (path, w, h, fps, total, output), flush=True)

    writer = None
    if args.save_video:
        video_out = os.path.splitext(os.path.splitext(output)[0])[0] + ".mp4"
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(tag + "annotated video -> " + video_out, flush=True)

    model = YOLO(args.model)
    reid = load_reid(args)
    track_state = {}   # track_id -> {"global_id", "embedding"}  (EMA prototype of that track)
    gallery = {}       # global_id -> prototype; kept after the track ends so re-entries can match
    next_gid = 1
    extra = {"half": True} if args.half else {}  # passing half at all warns every frame on 8.4+
    header = {
        "video": {"width": w, "height": h, "fps": float(fps), "total_frames": total,
                  "duration_sec": round(total / fps, 3), "source": os.path.basename(path)},
        "coordinate_system": COORDINATE_SYSTEM,
    }
    dump = lambda o: json.dumps(o, separators=(",", ":"))

    started, idx, written = time.perf_counter(), 0, 0
    # Header first, then one frame object at a time; "]}" closes it even if interrupted.
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write(dump(header)[:-1] + ',"frames":[')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                r = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, classes=[0],
                                device=args.device, imgsz=args.imgsz, verbose=False, **extra)[0]
                if idx % args.reid_interval == 0 and r.boxes is not None and r.boxes.id is not None:
                    visible = r.boxes.id.int().tolist()
                    fh, fw = frame.shape[:2]
                    tids, crops = [], []
                    for tid, (x1, y1, x2, y2) in zip(visible, r.boxes.xyxy.tolist()):
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(fw, int(x2)), min(fh, int(y2))
                        if x2 - x1 >= MIN_CROP_W and y2 - y1 >= MIN_CROP_H:
                            tids.append(tid)
                            crops.append(frame[y1:y2, x1:x2])
                    active = {track_state[t]["global_id"] for t in visible if t in track_state}
                    for tid, emb in zip(tids, reid_embed(reid, crops) if crops else []):
                        st = track_state.get(tid)
                        if st is None:  # new track: best gallery match among identities not on screen
                            best_id, best_sim = None, -1.0
                            for gid, proto in gallery.items():
                                sim = float(emb @ proto)
                                if gid not in active and sim > best_sim:
                                    best_id, best_sim = gid, sim
                            if best_sim >= args.reid_threshold:
                                st = {"global_id": best_id, "embedding": gallery[best_id]}
                            else:
                                st = {"global_id": next_gid, "embedding": emb}
                                next_gid += 1
                            track_state[tid] = st
                            active.add(st["global_id"])
                        # Established tracks keep their global_id; ReID only refreshes the prototype.
                        st["embedding"] = l2(0.8 * st["embedding"] + 0.2 * emb)
                        gallery[st["global_id"]] = st["embedding"]
                if writer is not None:
                    writer.write(r.plot())
                if idx % args.interval == 0:
                    rec = {"frame": idx, "time": round(idx / fps, 3), "persons": persons_of(r, args, track_state)}
                    f.write(("," if written else "") + dump(rec))
                    written += 1
                idx += 1
                if idx % 1000 == 0:
                    el = time.perf_counter() - started
                    if total:
                        print(tag + "%d/%d (%.1f%%) %.1f fps, eta %.0f min"
                              % (idx, total, 100.0 * idx / total, idx / el,
                                 (total - idx) / max(1e-6, idx / el) / 60), flush=True)
                    else:
                        print(tag + "%d frames, %.1f fps" % (idx, idx / el), flush=True)
        except KeyboardInterrupt:
            print(tag + "interrupted -- closing file")
        finally:
            cap.release()
            if writer is not None:
                writer.release()  # finalises the mp4 so it stays playable after Ctrl+C
            f.write("]}")

    el = time.perf_counter() - started
    print(tag + "done: %d frames read, %d written, %.1f s (%.1f fps), %s = %.1f MB"
          % (idx, written, el, idx / max(1e-6, el), output, os.path.getsize(output) / 1048576), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", default=default_inputs(),
                   help="one or more videos, processed concurrently (default: cameras.txt)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--model", default="yolo26m-pose.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default="0", help="'0' or 'cpu'")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--kpt-conf", type=float, default=0.30)
    p.add_argument("--view-fov", type=float, default=50.0)
    p.add_argument("--save-video", action="store_true", help="also write an annotated N_YYMMDD.mp4")
    p.add_argument("--interval", type=int, default=5, help="write every Nth frame (tracking still runs on all)")
    p.add_argument("--reid-interval", type=int, default=5, help="run ReID on all visible people every Nth frame")
    p.add_argument("--reid-threshold", type=float, default=0.6, help="min cosine similarity to reuse a global_id")
    p.add_argument("--reid-weights", default="solider_swin_small_msmt17.pth")
    p.add_argument("--reid-repo", default="SOLIDER-REID", help="clone of github.com/tinyvision/SOLIDER-REID")
    args = p.parse_args()
    if not args.input:
        p.error("no --input given and cameras.txt is missing or empty")
    args.interval = max(1, args.interval)
    args.reid_interval = max(1, args.reid_interval)
    for f in (args.reid_weights, os.path.join(args.reid_repo, "model", "backbones", "swin_transformer.py")):
        if not os.path.isfile(f):
            p.error("ReID file not found: " + f)

    os.makedirs(args.out_dir, exist_ok=True)
    date = datetime.now().strftime("%y%m%d")
    jobs = [(i, path, os.path.join(args.out_dir, "%d_%s.json.gz" % (i, date)))
            for i, path in enumerate(args.input, 1)]

    # Fetch the weights once up front, so concurrent workers don't race to download them.
    from ultralytics import YOLO
    YOLO(args.model)

    started = time.perf_counter()
    if len(jobs) == 1:
        run(*jobs[0], args)
    else:
        # One process per video: each owns its model and tracker state.
        procs = [mp.Process(target=run, args=(*job, args)) for job in jobs]
        for pr in procs:
            pr.start()
        try:
            for pr in procs:
                pr.join()
        except KeyboardInterrupt:
            for pr in procs:
                pr.join()
    print("total wall time: %.1f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
