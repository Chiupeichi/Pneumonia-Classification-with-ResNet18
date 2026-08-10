# Chest X-ray：肺炎分類與肺部分割

這個 repository 將原本的 Notebook 整理成兩個可重用、可測試的 PyTorch 專案：

- `classification/`：使用預訓練 ResNet-18 判斷 Normal / Pneumonia。
- `segmentation/`：使用 U-Net 產生肺部二值遮罩。

原始 Notebook 與既有 `.pth` 權重會保留在本機，方便比對與相容舊模型，但不會提交到公開 repository。`data/`、`db/` 與本機路徑相關的實驗檔案也不會上傳，以免公開影像、資料位置或病人層級 metadata。這是研究／教學用途的影像模型，不應直接用於臨床診斷。

## 主要改善

### Classification

- 訓練和推論共用相同的灰階三通道、尺寸與 ImageNet normalization，修正舊推論流程的 preprocessing mismatch。
- 自動檢查三個 split 的類別映射，避免標籤索引悄悄顛倒。
- 依訓練集類別數量加權 loss，改善常見的肺炎資料不平衡。
- 加入 AdamW、learning-rate scheduler、early stopping、CUDA mixed precision 與原子化 checkpoint。
- checkpoint 會保存 `class_to_idx` 和 preprocessing metadata；也相容原本只保存 state dict 的模型。

### Segmentation

- U-Net 統一輸出 logits，只在 loss／metric／推論處做 sigmoid，避免重複 sigmoid。
- mask resize 和 rotation 強制使用 nearest-neighbor，避免產生不存在的灰階標籤。
- 使用 BCE + Dice loss，並同時回報 Dice 與 IoU。
- 同時支援 `image.png`／`image.png` 與 `image.png`／`image_mask.png` 兩種配對方式，避免漏掉 China CXR masks。
- 固定 seed 且建立互斥的 train/validation split，結果可以重現。
- 預測 mask 會還原到原圖尺寸，也能另外輸出彩色 overlay。
- 可直接載入原本 `lung_unet_model.pth` 的 layer names。

## 安裝

建議使用 Python 3.10–3.12 建立獨立環境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Classification

資料夾需使用 `ImageFolder` 格式，三個 split 的類別名稱必須一致：

```text
chest_xray/
├── train/NORMAL/...
├── train/PNEUMONIA/...
├── val/NORMAL/...
├── val/PNEUMONIA/...
├── test/NORMAL/...
└── test/PNEUMONIA/...
```

訓練：

```bash
python -m classification.train \
  --data-dir /path/to/chest_xray \
  --output checkpoints/classification_best.pth
```

使用新 checkpoint 推論；若要使用舊權重，請自行放到 repository 根目錄或指定其本機路徑：

```bash
python -m classification.predict /path/to/xray.jpeg \
  --checkpoint pneumonia_resnet_model.pth
```

## Segmentation

影像和 mask 必須位於不同資料夾；mask 可使用完全相同檔名，或在副檔名前加 `_mask`：

```bash
python -m segmentation.train \
  --images '/path/to/CXR_png' \
  --masks '/path/to/masks' \
  --output checkpoints/segmentation_best.pth
```

輸出 mask 與 overlay：

```bash
python -m segmentation.predict /path/to/xray.png \
  --checkpoint lung_unet_model.pth \
  --output outputs/lung_mask.png \
  --overlay outputs/lung_overlay.png
```

`--device auto` 會依序選擇 CUDA、Apple Silicon MPS、CPU。若 DataLoader 在 Notebook 或 macOS 多程序環境出錯，維持預設 `--workers 0` 即可。

## 驗證

```bash
python -m compileall classification segmentation common.py
pytest -q
```

## 專案結構

```text
classification/   # ResNet model、data pipeline、train/eval、predict CLI
segmentation/     # U-Net model、paired dataset、loss/metrics、train/predict CLI
tests/            # reproducibility、shape、loss 與 metric 測試
data/             # 原始 metadata 與 split（路徑欄位需依本機資料位置更新）
db/               # 原始 SQLite metadata
*.ipynb           # 舊版 Notebook（只保留在本機，不上傳）
*.pth             # 模型權重（只保留在本機，不上傳）
```
