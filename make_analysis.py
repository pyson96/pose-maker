#!/usr/bin/env python
"""Minimal video -> gzipped analysis JSON (YOLO26l-Pose + ByteTrack + SOLIDER ReID), for bakery.html.

    python make_analysis.py --input a.mp4 b.mp4
        -> results/YYYY-MM/1_YYMMDD_HHMM.json.gz, 2_...   (inputs processed concurrently)
    python make_analysis.py --input rtsp://... --save-video
        -> 1_YYMMDD_HHMM.json.gz + .mp4      (Ctrl+C to stop; both files are closed cleanly)
    python make_analysis.py
        -> analyses the cameras listed in cameras.txt
    python make_analysis.py --input rtsp://cam1 rtsp://cam2 --overlap-file overlap_zones.json
        -> two cameras share global_ids through one overlap zone (drawn on first run if the file is missing)

Keypoints are a flat array in COCO-17 order: [x0, y0, conf0, x1, y1, conf1, ...].
track_id is ByteTrack's temporary id; global_id is a ReID identity that survives exits and re-entries
(null until the track's first ReID frame).
Frames are streamed to disk as they are produced, so memory stays flat. The file ends with "people":
one entry per global_id with its route (ground point once per second), first/last seen and dwell time.
"""

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import signal
import time
from datetime import datetime
from multiprocessing.managers import SyncManager

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
DEFAULT_HANDOFF_TTL = 5.0
DEFAULT_CROSS_REID_THRESHOLD = 0.70   # handoff_mode "appearance" only
# "position": someone who left the other camera's overlap zone within handoff_ttl is the same person,
# whatever they look like (the shared passage shows little more than a head). "appearance": also needs
# cross_reid_threshold similarity.
DEFAULT_HANDOFF_MODE = "position"
DEFAULT_ZONE_OVERLAP = 0.10   # a person is in the zone once this share of their box overlaps it
TRACK_GRACE_SEC = 1.0        # a track missed for less than this still holds its global_id (detection dropouts)
PATH_STEP_SEC = 1.0          # one route point per person per second in the "people" summary
RECONNECT_SEC = 5.0          # wait between attempts to reopen a dropped live stream
STREAM_TIMEOUT_MS = 10000    # RTSP open/read timeout, so a dead camera can't block past the stop time


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


def log(*parts):
    """print with a [HH:MM:SS] prefix, flushed at once so a redirected log file is current."""
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def open_capture(path):
    if path.lower().startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                                                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS])
    return cv2.VideoCapture(path)


def reopen_capture(path, should_stop, tag):
    """Keep trying to reopen a dropped live stream until it works or should_stop() -> capture or None."""
    attempt = 0
    while not should_stop():
        time.sleep(RECONNECT_SEC)
        attempt += 1
        cap = open_capture(path)
        if cap.isOpened():
            log(tag + "reconnected after %d attempt(s)" % attempt)
            return cap
        cap.release()
    return None


def load_overlap_config(path, n_cams=2):
    """overlap_zones.json -> dict with a "pairs" list, or None if the file is missing.

    Two layouts are read:
      * pairs (any number of cameras), cameras numbered by --input / cameras.txt order:
          {"pairs": [{"cameras": [1, 4], "polygons": [[[x, y], ...], [[x, y], ...]]}, ...], ...}
        polygons[k] is drawn on cameras[k]'s picture.
      * the original two-camera layout {"camera1": {"polygon": ...}, "camera2": {...}} = pair (1, 2).
    Pairs naming a camera that isn't running (e.g. a 4-camera file used with 2 inputs) are skipped.
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return normalize_overlap_config(cfg, n_cams, path)


def normalize_overlap_config(cfg, n_cams, path):
    if "pairs" not in cfg:
        for cam in ("camera1", "camera2"):
            if len(cfg.get(cam, {}).get("polygon", [])) < 3:
                raise SystemExit("%s: %s needs a polygon with at least 3 points" % (path, cam))
        cfg["pairs"] = [{"cameras": [1, 2], "polygons": [cfg["camera1"]["polygon"], cfg["camera2"]["polygon"]]}]
    pairs = []
    for p in cfg["pairs"]:
        a, b = p["cameras"]
        if a == b or len(p["polygons"]) != 2 or any(len(q) < 3 for q in p["polygons"]):
            raise SystemExit("%s: pair %s needs two different cameras and two polygons of >= 3 points" % (path, p["cameras"]))
        if max(a, b) > n_cams or min(a, b) < 1:
            log("overlap pair %d-%d skipped: only %d input(s)" % (a, b, n_cams))
            continue
        pairs.append({"cameras": [int(a), int(b)], "polygons": p["polygons"]})
    cfg["pairs"] = pairs
    cfg.setdefault("handoff_ttl", DEFAULT_HANDOFF_TTL)
    cfg.setdefault("cross_reid_threshold", DEFAULT_CROSS_REID_THRESHOLD)
    cfg.setdefault("handoff_mode", DEFAULT_HANDOFF_MODE)
    cfg.setdefault("zone_overlap_ratio", DEFAULT_ZONE_OVERLAP)
    if cfg["handoff_mode"] not in ("position", "appearance"):
        raise SystemExit("%s: handoff_mode must be \"position\" or \"appearance\"" % path)
    return cfg


def setup_overlap_config(paths, out_file):
    """Draw one overlap polygon per camera on a live preview frame (no YOLO / SOLIDER loaded)."""
    frames = []
    for path in paths:
        cap = open_capture(path)
        frame = None
        for _ in range(10):  # a few reads so an RTSP decoder settles on a full keyframe
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            raise SystemExit("overlap setup: cannot read a frame from " + path)
        frames.append(frame)
    return draw_overlap_config(frames, out_file)


def draw_overlap_config(frames, out_file):
    """Draw one overlap polygon per camera on the two given frames and save them.

    The frames must be at the analysis resolution: polygon points are stored in their pixels.
    Click inside a camera's panel to add a point to that camera's polygon (it becomes the selected one),
    r = reset the selected polygon, Enter = save both (>= 3 points each), Esc = cancel.
    Returns the saved config, or None when cancelled.
    """
    scales = [min(1.0, 800.0 / f.shape[1]) for f in frames]  # keep the window on screen
    views = [cv2.resize(f, None, fx=s, fy=s) for f, s in zip(frames, scales)]
    gap, bar = 12, 34
    offsets = [0, views[0].shape[1] + gap]
    polys, state = [[], []], {"cur": 0, "msg": ""}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for i, v in enumerate(views):
            if offsets[i] <= x < offsets[i] + v.shape[1] and y < v.shape[0]:
                state["cur"], state["msg"] = i, ""
                polys[i].append([int((x - offsets[i]) / scales[i]), int(y / scales[i])])

    win = "overlap setup"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    height = max(v.shape[0] for v in views)
    try:
        while True:
            canvas = np.full((height + bar, offsets[1] + views[1].shape[1], 3), 24, np.uint8)
            for i, v in enumerate(views):
                panel = v.copy()
                pts = (np.array(polys[i], np.float32).reshape(-1, 2) * scales[i]).astype(np.int32)
                if len(pts) >= 3:
                    fill = panel.copy()
                    cv2.fillPoly(fill, [pts], (0, 200, 255))
                    panel = cv2.addWeighted(fill, 0.25, panel, 0.75, 0)
                if len(pts):
                    cv2.polylines(panel, [pts], len(pts) >= 3, (0, 220, 255), 2, cv2.LINE_AA)
                    for q in pts:
                        cv2.circle(panel, (int(q[0]), int(q[1])), 4, (0, 0, 255), -1, cv2.LINE_AA)
                sel = i == state["cur"]
                cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                              (0, 255, 0) if sel else (90, 90, 90), 3 if sel else 1)
                label = "Camera %d  (%d pts)%s" % (i + 1, len(polys[i]), "  [selected]" if sel else "")
                cv2.rectangle(panel, (4, 4), (16 + 13 * len(label), 38), (0, 0, 0), -1)
                cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if sel else (220, 220, 220), 2, cv2.LINE_AA)
                canvas[:panel.shape[0], offsets[i]:offsets[i] + panel.shape[1]] = panel
            cv2.putText(canvas, state["msg"] or "click: add point | r: reset selected | Enter: save | Esc: cancel",
                        (10, height + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if state["msg"] else (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (ord("r"), ord("R")):
                polys[state["cur"]] = []
            if key in (13, 10):
                if all(len(q) >= 3 for q in polys):
                    break
                state["msg"] = "each camera needs at least 3 points"
    finally:
        cv2.destroyAllWindows()

    cfg = {"camera1": {"polygon": polys[0]}, "camera2": {"polygon": polys[1]},
           "handoff_ttl": DEFAULT_HANDOFF_TTL, "handoff_mode": DEFAULT_HANDOFF_MODE,
           "zone_overlap_ratio": DEFAULT_ZONE_OVERLAP,
           "cross_reid_threshold": DEFAULT_CROSS_REID_THRESHOLD}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log("overlap zones saved -> " + out_file)
    return cfg


def zone_coverage(polygon, w, h):
    """-> f(x1, y1, x2, y2): the share of that box lying inside the overlap polygon (0..1).

    In the shared passage a person is mostly hidden -- a head and some shoulder -- so the box's bottom
    edge is not their feet; how much of the box overlaps the zone says more. A summed-area table of the
    polygon mask makes each query four lookups, concave polygons included.
    """
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    sat = cv2.integral(mask)                                   # (h+1) x (w+1)

    def cover(x1, y1, x2, y2):
        x1, y1 = max(0, min(w, int(x1))), max(0, min(h, int(y1)))
        x2, y2 = max(0, min(w, int(x2))), max(0, min(h, int(y2)))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        return float(sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]) / area
    return cover


def allocate_global_id(shared, local_counter):
    """Next global_id; one counter shared by all camera processes when there are several."""
    if shared is None:
        local_counter[0] += 1
        return local_counter[0]
    with shared["lock"]:
        shared["next_gid"].value += 1
        return shared["next_gid"].value


def handoff_key(src, dst, gid):
    """Candidate "global_id gid, seen in src's zone shared with dst" -- one per camera pair direction."""
    return "%d>%d:%d" % (src, dst, gid)


def find_handoff_match(shared, camera, emb, now, ttl, threshold, active, partners, below=None):
    """Best recent overlap candidate handed to this camera -> (key, global_id, embedding, sim) or None.

    partners: the cameras whose shared zone this track is in (seen from this camera); only candidates
    those cameras published toward this one count.
    threshold None (handoff_mode "position"): whoever stood in the other camera's zone within ttl is
    taken whatever they look like -- in the passage only a head and a bit of shoulder show, so the
    appearance can't be trusted. Similarity then only picks between several candidates.
    below: a young track's own global_id -- only older (smaller) ids are considered, and nothing at all
    once the other camera already carries this id (the two cameras have agreed on it).
    """
    best, best_sim = None, -1.0
    cands = [c for c in shared["handoffs"].values()                       # one IPC round trip
             if c.get("to") == camera and c["camera"] in partners]
    if below is not None and any(c["global_id"] == below for c in cands):
        return None
    for cand in cands:
        key = handoff_key(cand["camera"], camera, cand["global_id"])
        if cand["global_id"] in active:
            continue
        if below is not None and cand["global_id"] >= below:
            continue
        dt = now - cand["last_seen"]
        if dt < 0 or dt > ttl:
            continue
        sim = float(emb @ cand["embedding"])
        if sim > best_sim:
            best, best_sim = (key, cand["global_id"], cand["embedding"], sim), sim
    return best if best is not None and (threshold is None or best_sim >= threshold) else None


def take_handoff(shared, key):
    """Consume a candidate so no second new track can take it; False if the other camera beat us to it."""
    with shared["lock"]:
        return shared["handoffs"].pop(key, None) is not None


def cleanup_handoffs(shared, camera, now, ttl):
    """Drop this camera's candidates whose TTL ran out."""
    for key, cand in shared["handoffs"].items():
        if cand["camera"] == camera and now - cand["last_seen"] > ttl:
            shared["handoffs"].pop(key, None)


def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the Manager outlives Ctrl+C until the workers closed their files


def update_route(routes, tid, t, x, y):
    """Record one sighting of a track at video time t (seconds), ground point (x, y)."""
    pt = [round(t, 3), round(x, 1), round(y, 1)]
    rt = routes.get(tid)
    if rt is None:
        routes[tid] = {"first": t, "last": t, "dwell": 0.0, "path": [pt], "end": pt}
        return
    if t - rt["last"] <= TRACK_GRACE_SEC:  # longer gaps (left the view, lost) are not dwell time
        rt["dwell"] += t - rt["last"]
    rt["last"], rt["end"] = t, pt
    if t - rt["path"][-1][0] >= PATH_STEP_SEC:
        rt["path"].append(pt)


def summarize_people(routes, track_state):
    """Merge track routes by final global_id -> list of {global_id, track_ids, first/last_seen, dwell, path}.

    Tracks that never got a global_id (gone before their first ReID) are listed on their own.
    """
    people = {}
    for tid, rt in sorted(routes.items(), key=lambda kv: kv[1]["first"]):
        gid = track_state[tid]["global_id"] if tid in track_state else None
        p = people.setdefault(gid if gid is not None else "t%d" % tid, {
            "global_id": gid, "track_ids": [], "first_seen": rt["first"], "last_seen": rt["last"],
            "dwell_sec": 0.0, "path": []})
        p["track_ids"].append(tid)
        p["first_seen"], p["last_seen"] = min(p["first_seen"], rt["first"]), max(p["last_seen"], rt["last"])
        p["dwell_sec"] += rt["dwell"]
        p["path"] += rt["path"] + ([rt["end"]] if rt["end"] is not rt["path"][-1] else [])
    out = []
    for p in people.values():
        p["path"].sort(key=lambda q: q[0])
        p["first_seen"], p["last_seen"] = round(p["first_seen"], 3), round(p["last_seen"], 3)
        p["span_sec"] = round(p["last_seen"] - p["first_seen"], 3)
        p["dwell_sec"] = round(p["dwell_sec"], 3)
        out.append(p)
    return sorted(out, key=lambda p: p["first_seen"])


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


def run(order, path, output, args, shared=None):
    """Analyse one video and stream its frames into `output` (gzip).

    shared (several inputs only): {"handoffs", "next_gid", "lock", "zones"} from the main process's Manager.
    order 1 -> camera1, order 2 -> camera2 in the overlap config.
    """
    from ultralytics import YOLO

    tag = "[%d] " % order
    cap = open_capture(path)
    if not cap.isOpened():
        cap.release()
        cap = None
        if args.deadline and path.lower().startswith("rtsp://"):
            # scheduled run: the camera may come up late, keep trying until the stop time
            log(tag + "cannot open %s -- retrying every %.0f s until %s" % (path, RECONNECT_SEC, args.until))
            cap = reopen_capture(path, lambda: time.time() >= args.deadline, tag)
        if cap is None:
            log(tag + "cannot open video: " + path)
            return
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:  # live streams often report 0 or a 90 kHz clock
        fps = 30.0
    total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # 0 for live streams
    live = total == 0
    log(tag + "%s: %dx%d @ %.3f fps, %d frames -> %s" % (path, w, h, fps, total, output))

    writer = None
    if args.save_video:
        video_out = os.path.splitext(os.path.splitext(output)[0])[0] + ".mp4"
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        log(tag + "annotated video -> " + video_out)

    model = YOLO(args.model)
    reid = load_reid(args)
    track_state = {}   # track_id -> {"global_id", "embedding" (EMA prototype), "in_overlap", "last_seen"}
    gallery = {}       # global_id -> prototype; kept after the track ends so re-entries can match
    gallery_seen = {}  # global_id -> last time seen; older than --id-keep and the id can't be reused
    local_gid = [0]    # global_id counter when this is the only camera
    routes = {}        # track_id -> sightings summary for the "people" section (see update_route)
    zones = shared.get("zones") if shared else None
    # this camera's side of every overlap pair it belongs to: [(other camera, coverage test)]
    my_zones = []
    for p in (zones or {}).get("pairs", []):
        if order in p["cameras"]:
            k = p["cameras"].index(order)
            my_zones.append((p["cameras"][1 - k], zone_coverage(np.array(p["polygons"][k], np.float32), w, h)))
    zone = my_zones or None
    if zone is not None:
        min_share = float(zones.get("zone_overlap_ratio", DEFAULT_ZONE_OVERLAP))
        ttl = float(zones["handoff_ttl"])
        by_position = zones.get("handoff_mode", DEFAULT_HANDOFF_MODE) == "position"
        cross_thr = None if by_position else float(zones["cross_reid_threshold"])
        handoffs = shared["handoffs"]
    extra = {"half": True} if args.half else {}  # passing half at all warns every frame on 8.4+
    header = {
        "video": {"width": w, "height": h, "fps": float(fps), "total_frames": total,
                  "duration_sec": round(total / fps, 3), "source": os.path.basename(path)},
        "coordinate_system": COORDINATE_SYSTEM,
        "camera_id": order,
        "overlap_enabled": zone is not None,
        "overlap_partners": [o for o, _ in my_zones],
    }
    dump = lambda o: json.dumps(o, separators=(",", ":"))

    started, idx, written = time.perf_counter(), 0, 0
    wall0 = time.time()
    header["started_at"] = datetime.fromtimestamp(wall0).isoformat(timespec="seconds")  # wall clock of time 0
    # Live streams drop and burst frames over a long day, so their "time" is wall-clock seconds since
    # started_at; files use the frame index (they are processed faster or slower than real time).
    header["time_base"] = "wall_clock" if live else "video"
    clock = (lambda: time.time() - wall0) if live else (lambda: idx / fps)

    def should_stop():
        return ((args.duration and time.perf_counter() - started >= args.duration)
                or (args.deadline and time.time() >= args.deadline))

    # Header first, then one frame object at a time; the per-person summary closes it even if interrupted.
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write(dump(header)[:-1] + ',"frames":[')
        try:
            while True:
                if should_stop():
                    log(tag + "stop time reached -- closing file")
                    break
                ok, frame = cap.read()
                if not ok:
                    if not live:
                        break
                    log(tag + "stream lost -- reconnecting every %.0f s" % RECONNECT_SEC)
                    cap.release()
                    cap = reopen_capture(path, should_stop, tag)
                    if cap is None:
                        log(tag + "stop time reached while reconnecting -- closing file")
                        break
                    continue
                now_t = clock()
                r = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, classes=[0],
                                device=args.device, imgsz=args.imgsz, verbose=False, **extra)[0]
                if r.boxes is not None and r.boxes.id is not None:
                    for tid, (x1, y1, x2, y2) in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                        update_route(routes, tid, now_t, (x1 + x2) / 2.0, y2)
                if idx % args.reid_interval == 0 and r.boxes is not None and r.boxes.id is not None:
                    # Handoff timestamps must be comparable across the two processes: wall clock for live
                    # streams, video time for files (assumes both files start at the same moment).
                    now = idx / fps if total else time.time()
                    visible = r.boxes.id.int().tolist()
                    fh, fw = frame.shape[:2]
                    tids, crops, boxes = [], [], []
                    for tid, (x1, y1, x2, y2) in zip(visible, r.boxes.xyxy.tolist()):
                        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(fw, int(x2)), min(fh, int(y2))
                        if x2 - x1 >= MIN_CROP_W and y2 - y1 >= MIN_CROP_H:
                            tids.append(tid)
                            crops.append(frame[y1:y2, x1:x2])
                            boxes.append((x1, y1, x2, y2))
                    for t in visible:
                        if t in track_state:
                            track_state[t]["last_seen"] = now
                    # identities on screen now or a moment ago: never hand them to a second track
                    active = {st["global_id"] for st in track_state.values() if now - st["last_seen"] <= TRACK_GRACE_SEC}
                    for tid, emb, box in zip(tids, reid_embed(reid, crops) if crops else [], boxes):
                        st = track_state.get(tid)
                        # in a zone = at least zone_overlap_ratio (10 %) of the person's box overlaps it;
                        # partners = the cameras whose shared zone that is
                        partners = {o for o, cover in my_zones if cover(*box) >= min_share}
                        inside = bool(partners)
                        if st is None:
                            # new track: 1) someone who just stood in the other camera's overlap zone (only
                            # when this track is in the zone too) -- by position alone in "position" mode,
                            # otherwise only if they look more alike than anyone in this camera's gallery;
                            # 2) this camera's gallery (identities not on screen); 3) a new global_id
                            # identities gone longer than --id-keep are forgotten: someone in similar
                            # clothes hours later is a new visitor, not the same one back
                            for gid in [g for g, t in gallery_seen.items() if now - t > args.id_keep]:
                                gallery.pop(gid, None)
                                gallery_seen.pop(gid, None)
                            best_id, best_sim = None, -1.0
                            for gid, proto in gallery.items():
                                sim = float(emb @ proto)
                                if gid not in active and sim > best_sim:
                                    best_id, best_sim = gid, sim
                            hit = (find_handoff_match(shared, order, emb, now, ttl, cross_thr, active, partners)
                                   if inside else None)
                            if (hit is not None and (by_position or hit[3] >= best_sim)
                                    and take_handoff(shared, hit[0])):
                                # by position the other camera's partial view says little about this one
                                st = {"global_id": hit[1], "embedding": emb if by_position else hit[2]}
                            elif best_sim >= args.reid_threshold:
                                st = {"global_id": best_id, "embedding": gallery[best_id]}
                            else:
                                # brand-new id: for handoff_ttl it may still yield to the other camera's
                                # older id (both cameras can see a person appear in the zone at once)
                                st = {"global_id": allocate_global_id(shared, local_gid), "embedding": emb,
                                      "fresh_until": now + ttl if zone is not None else -1.0}
                            track_state[tid] = st
                            active.add(st["global_id"])
                        elif inside and now <= st.get("fresh_until", -1.0):
                            hit = find_handoff_match(shared, order, emb, now, ttl, cross_thr, active, partners,
                                                     below=st["global_id"])
                            if hit is not None and take_handoff(shared, hit[0]):
                                old = st["global_id"]
                                for other, _ in my_zones:
                                    handoffs.pop(handoff_key(order, other, old), None)
                                gallery.pop(old, None)
                                gallery_seen.pop(old, None)
                                active.discard(old)
                                active.add(hit[1])
                                st["global_id"], st["fresh_until"] = hit[1], -1.0
                        # Established tracks keep their global_id; ReID only refreshes the prototype.
                        st["embedding"] = l2(0.8 * st["embedding"] + 0.2 * emb)
                        gallery[st["global_id"]] = st["embedding"]
                        gallery_seen[st["global_id"]] = now
                        if zone is not None:
                            # While in the zone the candidate stays fresh (the other camera usually sees the
                            # person before this one loses them); once seen outside the zone it is withdrawn.
                            # After leaving or vanishing inside the zone it lives on for handoff_ttl seconds.
                            # One candidate per partner camera whose shared zone the person is in.
                            for other, _ in my_zones:
                                key = handoff_key(order, other, st["global_id"])
                                if other in partners:
                                    handoffs[key] = {"global_id": st["global_id"], "camera": order, "to": other,
                                                     "embedding": st["embedding"].astype(np.float32), "last_seen": now}
                                elif other in st.get("in_overlap", ()):
                                    handoffs.pop(key, None)
                        st["in_overlap"], st["last_seen"] = partners, now
                    if zone is not None:
                        cleanup_handoffs(shared, order, now, ttl)
                if writer is not None:
                    writer.write(r.plot())
                if idx % args.interval == 0:
                    rec = {"frame": idx, "time": round(now_t, 3), "persons": persons_of(r, args, track_state)}
                    f.write(("," if written else "") + dump(rec))
                    written += 1
                idx += 1
                if idx % 1000 == 0:
                    el = time.perf_counter() - started
                    if total:
                        log(tag + "%d/%d (%.1f%%) %.1f fps, eta %.0f min"
                              % (idx, total, 100.0 * idx / total, idx / el,
                                 (total - idx) / max(1e-6, idx / el) / 60))
                    else:
                        log(tag + "%d frames, %.1f fps" % (idx, idx / el))
        except KeyboardInterrupt:
            log(tag + "interrupted -- closing file")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()  # finalises the mp4 so it stays playable after Ctrl+C
            people = summarize_people(routes, track_state)
            f.write('],"people":' + dump(people) + "}")

    el = time.perf_counter() - started
    log(tag + "done: %d frames read, %d written, %d people, %.1f s (%.1f fps), %s = %.1f MB"
          % (idx, written, sum(p["global_id"] is not None for p in people), el, idx / max(1e-6, el),
             output, os.path.getsize(output) / 1048576))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", default=default_inputs(),
                   help="one or more videos, processed concurrently (default: cameras.txt)")
    p.add_argument("--out-dir", default="results", help="results go to <out-dir>/YYYY-MM/")
    p.add_argument("--model", default="yolo26l-pose.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default="0", help="'0' or 'cpu'")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--kpt-conf", type=float, default=0.30)
    p.add_argument("--view-fov", type=float, default=50.0)
    p.add_argument("--save-video", action="store_true", help="also write an annotated N_YYMMDD_HHMM.mp4")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until the video ends / Ctrl+C)")
    p.add_argument("--until", default="", metavar="HH:MM",
                   help="stop cleanly at this time of day, e.g. 19:00 (for scheduled daily runs)")
    p.add_argument("--interval", type=int, default=5, help="write every Nth frame (tracking still runs on all)")
    p.add_argument("--reid-interval", type=int, default=5, help="run ReID on all visible people every Nth frame")
    p.add_argument("--reid-threshold", type=float, default=0.6, help="min cosine similarity to reuse a global_id")
    p.add_argument("--id-keep", type=float, default=3600,
                   help="seconds a global_id stays reusable after it was last seen (default 1 h)")
    p.add_argument("--reid-weights", default="solider_swin_small_msmt17.pth")
    p.add_argument("--reid-repo", default="SOLIDER-REID", help="clone of github.com/tinyvision/SOLIDER-REID")
    p.add_argument("--overlap-file", default="overlap_zones.json",
                   help="overlap zones for two cameras; drawn interactively and saved if missing ('' = off)")
    args = p.parse_args()
    if not args.input:
        p.error("no --input given and cameras.txt is missing or empty")
    args.deadline = 0.0
    if args.until:
        try:
            stop = datetime.combine(datetime.now().date(), datetime.strptime(args.until, "%H:%M").time())
        except ValueError:
            p.error("--until must be HH:MM, e.g. 19:00")
        if stop <= datetime.now():
            p.error("--until %s has already passed today" % args.until)
        args.deadline = stop.timestamp()
    args.interval = max(1, args.interval)
    args.reid_interval = max(1, args.reid_interval)
    for f in (args.reid_weights, os.path.join(args.reid_repo, "model", "backbones", "swin_transformer.py")):
        if not os.path.isfile(f):
            p.error("ReID file not found: " + f)

    now = datetime.now()
    out_dir = os.path.join(args.out_dir, now.strftime("%Y-%m"))  # one folder per month, e.g. results/2026-10
    os.makedirs(out_dir, exist_ok=True)
    if os.name == "nt":
        # Keep Windows from idle-sleeping mid-run (that drops the streams); released when the process exits.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    stamp = now.strftime("%y%m%d_%H%M")  # with the time, so a restart never overwrites
    jobs = [(i, path, os.path.join(out_dir, "%d_%s.json.gz" % (i, stamp)))
            for i, path in enumerate(args.input, 1)]

    # Overlap handoff between camera pairs; set it up before any model is loaded. With exactly two inputs
    # and no file yet the zones are drawn on the spot; more cameras need a "pairs" file (draw_zones.py).
    zones = None
    if args.overlap_file and len(jobs) >= 2:
        zones = load_overlap_config(args.overlap_file, len(jobs))
        if zones is None and len(jobs) == 2:
            log("%s not found -- draw the overlap zone on both cameras" % args.overlap_file)
            zones = setup_overlap_config(args.input, args.overlap_file)
            if zones is None:
                log("overlap setup cancelled -- running without cross-camera handoff")
            else:
                zones = normalize_overlap_config(zones, 2, args.overlap_file)
        elif zones is None:
            log("%s not found -- running without cross-camera handoff" % args.overlap_file)
        if zones is not None:
            log("overlap pairs: " + (", ".join("%d-%d" % tuple(p["cameras"]) for p in zones["pairs"]) or "none"))
            if not zones["pairs"]:
                zones = None

    # Fetch the weights once up front, so concurrent workers don't race to download them.
    from ultralytics import YOLO
    YOLO(args.model)

    started = time.perf_counter()
    if len(jobs) == 1:
        run(*jobs[0], args)
    else:
        # One process per video: each owns its model, tracker and ReID state. The Manager only holds the
        # shared global_id counter and the overlap handoff candidates (ids, timestamps, 768-d embeddings).
        manager = SyncManager()
        manager.start(ignore_sigint)
        shared = {"handoffs": manager.dict(), "next_gid": manager.Value("i", 0), "lock": manager.Lock(),
                  "zones": zones}
        procs = [mp.Process(target=run, args=(*job, args, shared)) for job in jobs]
        for pr in procs:
            pr.start()
        try:
            for pr in procs:
                pr.join()
        except KeyboardInterrupt:
            for pr in procs:
                pr.join()
        finally:
            manager.shutdown()
    log("total wall time: %.1f s" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
