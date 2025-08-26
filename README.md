# Multi‑Class Fish Classification

A simple, notebook‑driven project for training and evaluating image classifiers that recognize multiple fish species from images. This repo is optimized for **Google Colab** and local runs, with a single Jupyter notebook you can open and execute end‑to‑end.

---

## ✨ What’s inside

- **One clean notebook**: `MultiClass_Fish_Classifier (1).ipynb` — covers data loading, basic preprocessing, model training, evaluation, and inference.
- **Minimal repo**: easy to fork, duplicate, or run in Colab.
- **Bring‑your‑own data**: works with any folder‑structured dataset (one subfolder per class).

> Tip: Keep the repository light. Store large datasets and trained models in cloud storage (Drive/S3/HF Datasets) and mount or download at runtime.

---

## 📁 Repository Structure

```
.
├── MultiClass_Fish_Classifier (1).ipynb   # Main notebook (training + evaluation + inference)
└── README.md                               # You are here
```

---

## 🚀 Getting Started

### Option A — Run in Google Colab (recommended)
1. Open the notebook in Colab:
   - From GitHub: open `MultiClass_Fish_Classifier (1).ipynb` → “Open in Colab” (or copy the raw URL into Colab).
2. Set runtime to **GPU** (Runtime → Change runtime type → Hardware accelerator → GPU).
3. Run the cells top‑to‑bottom. Follow the instructions in the notebook to mount storage and set dataset paths.

### Option B — Run locally
1. Clone the repo and enter the folder:
   ```bash
   git clone https://github.com/thedynasty23/Multi-Class-Fish-Classification.git
   cd Multi-Class-Fish-Classification
   ```
2. Create a virtual environment (optional but recommended) and install dependencies you use in the notebook (e.g., TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit‑learn, Pillow, etc.). Example:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install tensorflow numpy pandas scikit-learn matplotlib pillow tqdm ipykernel
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   pip install notebook
   jupyter notebook
   ```

---

## 🗂️ Dataset Setup

The notebook is dataset‑agnostic. Use **any image dataset** organized like:

```
/data_root/
    train/
        class_a/
        class_b/
        ...
    val/
        class_a/
        class_b/
        ...
    test/               # optional
        class_a/
        class_b/
        ...
```

- You can point the notebook to Google Drive, a local path, or a downloaded zip you extract at runtime.
- If your dataset isn’t split, many Keras utilities (or a quick Python script) can create **train/val** splits for you.

---

## 🧠 Typical Workflow (inside the notebook)

1. **Imports & Config**: set seeds, paths, image size, batch size, epochs.
2. **Data Loading**: load images via generators or `tf.data`.
3. **Preprocessing & Augmentation**: rescaling, flips, rotations, zooms, etc.
4. **Model**: start with a compact CNN or a transfer‑learning backbone (e.g., MobileNetV2 / EfficientNet) — up to you.
5. **Training**: monitor metrics; use callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
6. **Evaluation**: accuracy, classification report, confusion matrix.
7. **Inference**: single image / batch predictions.
8. **Save Artifacts**: export the best model (`.h5` / SavedModel) and metrics plots.

> If you plan to commit outputs, save them under `outputs/` (ignored by default if you add a `.gitignore`).

---

## 📊 Suggested Visuals

If you generate plots during training, consider saving:
- `outputs/history_accuracy.png`
- `outputs/history_loss.png`
- `outputs/confusion_matrix.png`
- `outputs/sample_predictions.png`

Then embed them here:
```markdown
![Accuracy](outputs/history_accuracy.png)
![Confusion Matrix](outputs/confusion_matrix.png)
```

---

## ⚙️ Configuration Snippets (optional)

**Train/Val split from a single folder**
```python
import os, shutil, random, pathlib
random.seed(42)

src = pathlib.Path("/path/to/images_by_class")
dst = pathlib.Path("/path/to/dataroot")
for split in ["train", "val"]:
    for c in os.listdir(src):
        (dst/split/c).mkdir(parents=True, exist_ok=True)

ratio = 0.2  # 20% for val
for c in os.listdir(src):
    imgs = list((src/c).glob("*.jpg"))
    random.shuffle(imgs)
    k = int(len(imgs)*ratio)
    for p in imgs[:k]:
        shutil.copy(p, dst/"val"/c/p.name)
    for p in imgs[k:]:
        shutil.copy(p, dst/"train"/c/p.name)
```

**Keras data loaders**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (224, 224)
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                               height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory("/data_root/train", target_size=img_size, batch_size=32, class_mode="categorical")
val = val_gen.flow_from_directory("/data_root/val", target_size=img_size, batch_size=32, class_mode="categorical")
```

---

## 🤝 Contributing

PRs and issues are welcome. If you add utilities (e.g., helpers to download datasets, evaluation scripts), keep the notebook tidy and document your changes.

---

## 📄 License

Add a `LICENSE` file (e.g., MIT) for clarity if you plan to share/extend this work.

---

## 🙌 Acknowledgements

Thanks to open‑source libraries and the community. If your work is based on a specific dataset/challenge, credit it in this README.
