# LiPAD YOLO Training Lab

Standalone training workspace for comparing **YOLOv8**, **YOLOv11**, and **YOLOv12** segmentation models.

Located at: `PROJECT_LIPAD_ANALYSIS/LIPAD_YOLO_TRAINING` (sibling to `Corrosion/`).

## Structure

```
LIPAD_YOLO_TRAINING/
├── shared/                 # Model registry + unified trainer
├── crack_detection/        # Crack YOLO seg training
│   ├── datasets/           # Your images + labels
│   ├── train_yolov8.py
│   ├── train_yolov11.py
│   ├── train_yolov12.py
│   ├── train_all.py
│   └── colab_train.ipynb   # Google Colab / Cursor notebook
├── corrosion_detection/    # Corrosion YOLO seg training (replaces HSV pipeline)
│   └── (same layout)
├── requirements.txt
└── setup.ps1
```

## Local setup (Windows)

```powershell
cd C:\Users\ADMIN\PROJECT_LIPAD_ANALYSIS\LIPAD_YOLO_TRAINING
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

## Train crack models

```powershell
python crack_detection/train_yolov8.py
python crack_detection/train_yolov11.py
python crack_detection/train_yolov12.py
# or
python crack_detection/train_all.py
```

## Train corrosion models

```powershell
python corrosion_detection/train_yolov8.py
python corrosion_detection/train_all.py
```

## Unified CLI (from repo root)

```powershell
python -m shared.trainer --task crack --model yolov11 --epochs 100 --batch 8
python -m shared.trainer --task corrosion --model yolov12 --device 0
```

## Google Colab + Cursor notebooks

Open either notebook in **Cursor** (built-in Jupyter) or upload to **Google Colab**:

- `crack_detection/colab_train.ipynb`
- `corrosion_detection/colab_train.ipynb`

Install the recommended Jupyter extension when prompted (`.vscode/extensions.json`).

## Deploy weights to LiPAD app

Copy best weights after training:

```
crack_detection/runs/yolov11/crack_yolov11_seg/weights/best.pt
→ Corrosion/PROJECT_LIPAD/models/best.pt
```

Update `main_app.py` / runtime engine `--weights` path when comparing models.
