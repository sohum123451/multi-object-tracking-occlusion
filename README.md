# Multi-Object Tracking Evaluation

Evaluation framework comparing tracking algorithms on standard MOT17 benchmark sequences.

## Repository Layout

```text
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── MOT17/
│       └── train/
│           ├── MOT17-02-DPM/
│           ├── MOT17-02-FRCNN/
│           ├── MOT17-02-SDP/
│           ├── MOT17-04-DPM/
│           ├── ...
│           └── MOT17-13-SDP/
├── results/
├── scripts/
│   ├── download_mot17.py
│   ├── run_detect.py
│   ├── run_track.py
│   ├── evaluate.py
│   ├── analyze_failures.py
│   ├── render_video.py
│   └── export_web.py
├── src/
│   └── tracking/
│       ├── __init__.py
│       ├── detect.py
│       ├── kalman.py
│       ├── sort.py
│       ├── deepsort.py
│       ├── embed.py
│       └── io.py
├── third_party/
│   ├── README.md
│   └── TrackEval/
└── web/
    └── index.html
```

## Third Party Dependencies

TrackEval is vendored under `third_party/TrackEval` from `https://github.com/JonathonLuiten/TrackEval` (MIT License). See `third_party/README.md` and `third_party/TrackEval/LICENSE` for details.

## Dataset Preparation

To download and extract the MOT17 train split annotations and verify the directory structure:

```bash
python scripts/download_mot17.py
```

Each sequence is verified against the required MOTChallenge directory structure:
- `seqinfo.ini`: sequence metadata configuration
- `img1/`: frame image directory
- `gt/gt.txt`: ground truth bounding box annotations
- `det/`: public detections directory containing `det.txt`
