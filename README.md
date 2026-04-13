# Real-Time Multi-Object Tracking System

This project detects people in real time with YOLOv8 and tracks them with DeepSORT plus ReID embeddings. It supports webcam input or a video file, displays tracking boxes and IDs, shows FPS, and writes tracking metadata to CSV.

## Features

- YOLOv8 person detection
- DeepSORT multi-object tracking with ReID
- Real-time FPS display
- Webcam or video-file input
- CSV logging of track history
- Optional CUDA acceleration when available

## Run

```bash
python main.py
```

To use a video file, update `config/config.yaml`:

```yaml
video_source: path/to/video.mp4
```

## GitHub setup

1. Initialize the repository.
2. Commit the source files.
3. Create a new empty GitHub repository.
4. Add the GitHub remote and push the `main` branch.
