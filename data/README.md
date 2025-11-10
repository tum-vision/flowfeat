# 📁 ./data

This directory contains **symbolic links** to external datasets required for training and evaluation.

> Please **download them manually** from the official sources below and create **symlinks** pointing to their locations on your local system.

---

## Suggested Datasets

### 1. [DAVIS 2017](https://davischallenge.org/davis2017/code.html) (used for evaluation)

- **Description:** A benchmark dataset for video object segmentation (training and validation).
- **Official download page:** [https://davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html)
- **Expected directory name:** `davis2017`

---

### 2. [YouTube-VOS 2019](https://youtube-vos.org/dataset/) (used for training)

- **Description:** A large-scale dataset for video object segmentation.
- **Official download page:** [https://youtube-vos.org/dataset/](https://youtube-vos.org/dataset/)
- **Expected directory name:** `ytvos`

---

## 🔗 Creating Symbolic Links

Once the datasets are downloaded and extracted, create **symlinks** inside this folder (`./data/`) so that your training and evaluation scripts can find them.

### Linux / macOS

```bash
# Navigate to the repository's data directory
cd ./data

# Link DAVIS 2017
ln -s /path/to/datasets/DAVIS-2017 ./davis2017

# Link YouTube-VOS 2019
ln -s /path/to/datasets/YouTubeVOS-2019 ./ytvos
```

Verify:

```
./data/davis2017/Annotations/
./data/davis2017/ImageSets/
./data/davis2017/JPEGImages/
./data/davis2017/README.md
./data/davis2017/SOURCES.md
```

```
./data/ytvos/train/
./data/ytvos/train_all_frames/
./data/ytvos/valid/
./data/ytvos/README
```
