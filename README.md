# pose-maker

CCTV 영상(파일 또는 RTSP)에서 사람의 자세·위치·시선 방향을 분석해 `.json.gz`로 저장하고, HTML 뷰어로 확인하는 도구입니다.

- **포즈 추정**: YOLO26-Pose (COCO 17 키포인트)
- **추적**: ByteTrack — 화면 안에서의 임시 `track_id`
- **재식별(ReID)**: SOLIDER Swin-Small — 화면을 나갔다 다시 들어와도 유지되는 `global_id`

## 요구 사항

- Windows / Linux, Python 3.10 이상
- NVIDIA GPU + CUDA 12.1 지원 드라이버 (CPU도 가능하지만 매우 느림)

## 설치

### 1. 코드 받기와 가상환경

```bash
git clone https://github.com/pyson96/pose-maker.git
cd pose-maker
python -m venv venv
venv\Scripts\activate          # Linux: source venv/bin/activate
```

### 2. 패키지 설치

torch는 반드시 **CUDA 인덱스에서 먼저** 설치하세요. 순서를 바꾸면 CPU 버전 torch가 설치됩니다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True`가 나와야 GPU를 사용합니다.

### 3. git에 포함되지 않는 파일 준비

용량 때문에 아래 파일은 저장소에 없습니다(`.gitignore`). 직접 준비해야 합니다.

| 파일 | 준비 방법 |
|---|---|
| `yolo26l-pose.pt` | 첫 실행 시 자동 다운로드 — 할 일 없음 |
| `SOLIDER-REID/` | `git clone https://github.com/tinyvision/SOLIDER-REID` |
| `solider_swin_small_msmt17.pth` (약 199MB) | 기존 PC에서 복사하거나, SOLIDER-REID README의 MSMT17 Swin-Small 가중치 링크에서 다운로드 |
| `cameras.txt` | `cameras.example.txt`를 복사해 계정/비밀번호 입력 (비밀번호가 있어 git 제외) |

`SOLIDER-REID/`와 `.pth`가 프로젝트 폴더 바로 아래에 있어야 합니다. 없으면 `ReID file not found` 오류로 멈춥니다.

기존 PC에서 SSH로 복사하는 예:

```bash
scp solider_swin_small_msmt17.pth <user>@<host>:<pose-maker 경로>/
```

`mmcv`는 설치하지 않아도 됩니다(코드에서 대체 처리).

준비가 끝난 폴더 구조:

```
pose-maker/
├── make_analysis.py
├── screenshot.py
├── requirements.txt
├── bakery.html, office.html
├── cameras.txt
├── solider_swin_small_msmt17.pth
└── SOLIDER-REID/
    └── model/backbones/swin_transformer.py
```

## 사용법

### 카메라 스크린샷 — `screenshot.py`

카메라 연결과 화각을 확인할 때 씁니다. 각 카메라의 현재 화면을 현재 폴더에 `screen1.jpg`, `screen2.jpg`로 저장합니다.

```bash
python screenshot.py                       # 기본 카메라 2대
python screenshot.py --input <주소1> <주소2>  # 다른 주소
```

### 영상 분석 — `make_analysis.py`

```bash
python make_analysis.py                          # 기본 카메라 2대 동시 분석
python make_analysis.py --input a.mp4 b.mp4      # 영상 파일 분석
python make_analysis.py --save-video             # 분석 결과를 그린 mp4도 함께 저장
```

- 입력마다 별도 프로세스로 동시에 처리합니다.
- 결과는 `1_YYMMDD_HHMM.json.gz`, `2_YYMMDD_HHMM.json.gz` 형식으로 저장됩니다(번호는 입력 순서, 시각은 시작 시각). 같은 날 다시 실행해도 덮어쓰지 않습니다.
- RTSP는 끝이 없으므로 **Ctrl+C**, `--duration`, `--until`로 종료합니다. 어느 방법이든 json.gz와 mp4는 정상적으로 닫힙니다.
- RTSP 연결이 끊기면 5초마다 다시 접속을 시도합니다. 종료 시각이 되면 재접속을 멈추고 파일을 닫습니다.

자주 쓰는 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input` | `cameras.txt` | 영상 파일 또는 RTSP 주소, 여러 개 가능 |
| `--out-dir` | `.` | 결과 저장 폴더 |
| `--model` | `yolo26l-pose.pt` | 더 빠르게 하려면 `yolo26m-pose.pt` / `yolo26n-pose.pt` |
| `--device` | `0` | GPU 번호 또는 `cpu` |
| `--interval` | `5` | N프레임마다 기록(추적은 모든 프레임에서 수행) |
| `--reid-interval` | `5` | N프레임마다 ReID 실행 |
| `--reid-threshold` | `0.6` | 같은 사람으로 볼 최소 유사도 |
| `--save-video` | 끔 | 분석 결과를 그린 영상 저장 |
| `--duration` | `0` | N초 후 자동 종료 (0 = 영상 끝 또는 Ctrl+C까지) |
| `--until` | 없음 | 지정한 시각(`HH:MM`)에 자동 종료, 예: `--until 19:00` |

전체 옵션은 `python make_analysis.py -h`로 확인하세요.

### 기본 카메라 주소 변경

기본 카메라 주소는 `cameras.txt`에 한 줄에 하나씩 적습니다(`#`으로 시작하면 주석). `make_analysis.py`와 `screenshot.py`가 모두 이 파일을 씁니다. `--input`을 주지 않았는데 이 파일이 없으면 오류로 멈춥니다.

```bash
copy cameras.example.txt cameras.txt     # Linux: cp
```

비밀번호가 들어 있으므로 `cameras.txt`는 git에 올리지 마세요(`.gitignore`에 포함).

### 매일 자동 실행 (10:00~19:00)

`run_daily.bat`이 `make_analysis.py --until 19:00`을 실행하고, 로그를 `logs\analysis.log`에 이어 붙입니다. Windows 작업 스케줄러에 등록합니다.

```bat
schtasks /create /tn PoseMaker /tr "C:\Users\<user>\pose-maker\run_daily.bat" /sc daily /st 10:00 /et 19:15 /k
```

- `/st 10:00`: 매일 10시에 시작합니다.
- 19시 종료는 프로그램이 `--until 19:00`으로 **스스로** 합니다. 그래야 결과 파일이 정상적으로 닫힙니다.
- `/et 19:15 /k`: 19:15까지 끝나지 않으면 강제 종료하는 안전장치입니다. 강제 종료되면 그날 파일은 깨지므로, 19시 정각으로 걸지 마세요.
- 위 명령은 로그인한 사용자 세션에서만 실행됩니다. 로그아웃 상태에서도 돌리려면 `/ru <계정> /rp <비밀번호>`를 추가합니다.
- `overlap_zones.json`을 먼저 만들어 두세요. 없으면 영역 그리기 창이 떠서 자동 실행이 멈춥니다.

## 결과 보기

브라우저로 HTML 파일을 열고 `.json.gz`를 불러오거나 창에 끌어다 놓습니다.

- `office.html` — Office CCTV 디지털 트윈. 원본 영상을 함께 열면 배경이 영상으로 바뀝니다.
- `bakery.html` — Bakery CCTV 3D 재구성

## 출력 형식

```jsonc
{
  "video": { "width", "height", "fps", "total_frames", "duration_sec", "source" },
  "coordinate_system": { ... },   // 원점 좌상단, 픽셀 단위
  "camera_id": 1, "overlap_enabled": true,
  "started_at": "2026-10-02T18:01:00",   // time 0의 실제 시각
  "time_base": "wall_clock",              // RTSP: time = started_at부터 흐른 실제 초 / 파일: "video" (프레임 번호 ÷ fps)
  "frames": [
    { "frame": 0, "time": 0.0, "persons": [
      { "track_id", "global_id", "bbox": {"x1","y1","x2","y2"}, "det_conf",
        "ground_point": [x, y],     // bbox 하단 중앙 (발 위치)
        "keypoints": [x0, y0, c0, x1, y1, c1, ...],   // COCO-17 순서
        "head": { "origin", "direction", "angle_deg", "source", "fov_deg" } }
    ]}
  ],
  "people": [                      // 사람별 요약 (파일을 닫을 때 기록)
    { "global_id": 2, "track_ids": [1, 4],
      "first_seen": 0.4, "last_seen": 19.8,   // 처음/마지막으로 보인 시각 (초, frames의 time과 같은 기준)
      "span_sec": 19.4,                        // last_seen - first_seen
      "dwell_sec": 17.2,                       // 실제로 화면에 보인 시간의 합
      "path": [[t, x, y], ...] }               // 동선: 1초마다 발 위치 + 마지막 위치
  ]
}
```

- `track_id`: ByteTrack 임시 ID. 사람이 화면을 나가면 바뀝니다.
- `global_id`: ReID로 부여한 ID. 다시 들어와도 유지됩니다. 첫 ReID 전에는 `null`입니다.
- `head`: 코와 귀·눈·어깨 위치로 추정한 머리 방향. 추정할 수 없으면 `null`입니다.
- `people`: 같은 `global_id`의 track들을 합친 사람별 동선과 체류시간입니다.
  - `dwell_sec`: 1초 이상 안 보인 구간은 빼고 계산합니다. 화면을 나갔다 다시 들어온 시간은 `span_sec`에는 들어가지만 `dwell_sec`에는 들어가지 않습니다.
  - 카메라마다 파일이 따로 있으므로, 두 카메라에 걸친 전체 동선은 같은 `global_id`의 두 파일 항목을 합쳐서 보면 됩니다.
  - 첫 ReID 전에 사라진 짧은 track은 `global_id: null`로 따로 나옵니다.
