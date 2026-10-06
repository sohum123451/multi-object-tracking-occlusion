# Multi-Object Tracking Evaluation: SORT vs DeepSORT

This project evaluates and compares SORT and DeepSORT multi-object tracking pipelines on standard MOT17 benchmark sequences. It provides an end-to-end framework covering YOLOv8 detection, Kalman filtering, OSNet appearance re-identification, TrackEval benchmarking, parameter ablation, failure analysis, video rendering, and a 3D space-time web visualizer.

## System Requirements

- Python 3.10 or newer (tested with Python 3.10 and 3.12).
- Node.js 18 or newer and npm (for the web visualizer).
- Hardware: CPU execution is supported across all modules; NVIDIA GPU with CUDA is optional and used automatically if detected.
- Dependencies: exact package versions are pinned in `requirements.txt`.

## Environment Setup

Install the required Python packages and web visualizer dependencies:

```bash
pip install -r requirements.txt
cd web && npm install && cd ..
```

TrackEval is vendored directly under `third_party/TrackEval` (MIT License) with NumPy 2.x compatibility shims configured automatically.

## Pipeline Reproduction

Run each step using the following commands:

### 1. Download Dataset
Downloads MOT17 train split sequences and ground truth annotations from the Hugging Face dataset mirror:

```bash
python scripts/download_mot17.py
```

### 2. Pedestrian Detection
Runs YOLOv8 nano (`yolov8n.pt`) on class 0 (person), caching MOT-formatted detections to `results/dets/`:

```bash
python scripts/run_detect.py --seq MOT17-04 MOT17-02 MOT17-09 --weights yolov8n.pt --conf 0.25 --imgsz 640
```

### 3. Multi-Object Tracking
Runs SORT (Kalman filter + IoU association) and DeepSORT (Kalman filter + OSNet appearance matching cascade):

```bash
python scripts/run_track.py --tracker sort --seq MOT17-04 MOT17-02 MOT17-09
python scripts/run_track.py --tracker deepsort --seq MOT17-04 MOT17-02 MOT17-09
```

### 4. Benchmark Evaluation
Executes TrackEval against MOT17 ground truth and generates the primary metrics summary:

```bash
python scripts/evaluate.py --tracks-dir results/tracks --trackers sort deepsort --seq MOT17-02 MOT17-04 MOT17-09 --output-csv results/metrics.csv
```

### 5. Parameter Ablation
Sweeps detector confidence thresholds (0.3, 0.4, 0.5) and maximum track age (15, 30, 60 frames):

```bash
python scripts/run_ablation.py --seq MOT17-02 MOT17-04 MOT17-09 --confs 0.3 0.4 0.5 --max-ages 15 30 60 --trackers sort deepsort --output-csv results/ablation.csv
```

### 6. Failure Analysis
Identifies identity switches, classifies failure causes, produces cause distribution charts, and saves image pairs:

```bash
python scripts/analyze_failures.py --seq MOT17-02 MOT17-04 MOT17-09 --trackers sort deepsort --visibility-threshold 0.6 --occlusion-iou-threshold 0.2 --speed-threshold 15.0 --appearance-threshold 0.70 --output-csv results/id_switches.csv --figures-dir results/figures
```

### 7. Comparison Video Rendering
Renders a 300-frame side-by-side video of MOT17-04 comparing SORT and DeepSORT with 40-frame fading trajectory trails and identity switch rings:

```bash
python scripts/render_video.py --seq MOT17-04 --num-frames 300 --output-video results/figures/clip_sort_vs_deepsort.mp4
```

### 8. Space-Time Web Visualizer
Exports trajectory data, keyframes, and metadata, then launches or builds the Vite + Three.js + anime.js v4 visualizer:

```bash
python scripts/export_web.py --seq MOT17-04 --num-frames 300
cd web && npm run dev
```

To create a static production distribution for GitHub Pages:

```bash
cd web && npm run build
```

## Running Evaluation Alone

To re-evaluate existing tracker prediction files without rerunning detection or tracking:

```bash
python scripts/evaluate.py --tracks-dir results/tracks --trackers sort deepsort --seq MOT17-02 MOT17-04 MOT17-09 --output-csv results/metrics.csv
```

The script parses ground truth files in `data/MOT17/train`, calculates MOTA, MOTP, IDF1, HOTA, IDSW, FP, FN, Frag, MT, and ML, prints a formatted table, and updates `results/metrics.csv`.

## Results Directory Structure

All generated outputs are organized into predictable paths:

```text
results/
├── dets/
│   ├── MOT17-02.txt               # YOLOv8 detections in MOT format
│   ├── MOT17-02_meta.json         # Detection counts and runtime
│   ├── MOT17-04.txt
│   ├── MOT17-04_meta.json
│   ├── MOT17-09.txt
│   └── MOT17-09_meta.json
├── tracks/
│   ├── sort/
│   │   ├── MOT17-02.txt           # SORT tracking predictions
│   │   ├── MOT17-04.txt
│   │   └── MOT17-09.txt
│   └── deepsort/
│       ├── MOT17-02.txt           # DeepSORT tracking predictions
│       ├── MOT17-04.txt
│       └── MOT17-09.txt
├── embeddings/
│   ├── MOT17-02_osnet_x0_25.npz   # Cached L2-normalized 512-d OSNet embeddings
│   ├── MOT17-04_osnet_x0_25.npz
│   └── MOT17-09_osnet_x0_25.npz
├── metrics.csv                    # TrackEval benchmark metrics
├── ablation.csv                   # Grid search ablation metrics
├── id_switches.csv                # Catalog of all detected identity switches
└── figures/
    ├── clip_sort_vs_deepsort.mp4  # Side-by-side rendered comparison video
    ├── id_switch_causes.png       # Cause breakdown bar chart
    └── worst_switches/            # Image pairs for top 10 identity switches
        ├── worst_switch_01_sort_MOT17-09_f223.jpg
        └── ...
```

Web application distribution outputs are located in:

```text
web/
├── public/
│   ├── tracks_sort.json           # Footpoint coordinates and timestamps
│   ├── tracks_deepsort.json       # DeepSORT footpoint trajectories
│   ├── id_switches.json           # Annotated switch events
│   ├── meta.json                  # Sequence dimensions and frame counts
│   └── keyframes/                 # Downsampled video background textures
└── dist/                          # Static production bundle (index.html, JS, CSS)
```

## Failure Analysis Classification Thresholds

The failure analysis script `scripts/analyze_failures.py` inspects every instance where a ground truth pedestrian matches a different predicted track ID across consecutive matched frames (using IoU >= 0.5). Each event is classified into one or more categories using the following thresholds:

1. **Occlusion (`--visibility-threshold 0.6`, `--occlusion-iou-threshold 0.2`)**:
   - *Visibility rationale*: The MOT17 ground truth annotation provides visibility ratios between 0.0 and 1.0 based on manual annotation. A visibility value below 0.6 indicates that over 40% of the pedestrian bounding box is obscured by background scenery, street furniture, or other pedestrians. High visual occlusion degrades feature extraction and leads to spatial bounding box truncation.
   - *Inter-target overlap rationale*: An intersection-over-union of at least 0.2 between two ground truth bounding boxes indicates that the targets are physically overlapping in the image projection, causing detector bounding box merging or Hungarian assignment contention.
2. **Fast Motion (`--speed-threshold 15.0` px/frame)**:
   - *Kinematic rationale*: The linear constant-velocity Kalman filter assumes steady inter-frame displacement. A ground truth displacement exceeding 15 pixels per frame deviates significantly from normal pedestrian walking speeds at standard frame rates or reflects abrupt camera panning. This discrepancy drives the predicted state outside the spatial association gate.
3. **Similar Appearance (`--appearance-threshold 0.70` cosine similarity)**:
   - *Re-identification rationale*: Torchreid OSNet features are L2-normalized 512-dimensional descriptors. Cosine similarities exceeding 0.70 between the mean embeddings of two distinct ground truth pedestrians indicate nearly identical color and texture representations (such as matching dark coats or uniform attire), leading to ambiguous distance metrics during appearance cascade matching.
4. **Other**:
   - Assigned to any identity switch where visibility remained above 0.6, inter-target IoU remained below 0.2, displacement was below 15.0 px/frame, and appearance similarity was below 0.70. These events are primarily driven by detector false negatives or Kalman covariance inflation over unobserved frames.
