# classification_covid

COVID-19 chest X-ray classification with transfer learning: comparison of VGG19, DenseNet121 and MobileNetV2 (TensorFlow / Keras)

# COVID-19 Chest X-ray Classification: VGG19 vs DenseNet121 vs MobileNetV2

Comparison of three pretrained CNN architectures for classifying chest X-ray images into **4 classes**: COVID-19, Normal, Lung Opacity and Viral Pneumonia. Everything is implemented in TensorFlow / Keras and trained on a Tesla P100 GPU (Kaggle).

## Overview

| Project | Domain | Task | Models compared |
| ------- | ------ | ---- | --------------- |
| COVID-19 classification | Medical imaging (chest X-ray) | 4-class classification | VGG19, DenseNet121, MobileNetV2 |

## Dataset

**Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle)

| Class | Images |
| ----- | ------ |
| Normal | 10,192 |
| Lung Opacity | 6,012 |
| COVID | 3,616 |
| Viral Pneumonia | 1,345 |
| **Total** | **21,165** |

- Stratified split 80 / 20 (`random_state=42`): 16,932 training images and 4,233 validation images
- Images resized to 224x224, RGB
- Preprocessing: pixel rescaling to [0, 1] for VGG19 and DenseNet121, `preprocess_input` of MobileNetV2 for MobileNetV2
- The dataset is imbalanced (Normal is about 8x larger than Viral Pneumonia); no rebalancing was applied

## Training Setup

Two-stage transfer learning for each model:

1. **Feature extraction:** ImageNet-pretrained backbone frozen, new head trained (GlobalAveragePooling2D, Dense 128 + ReLU, Dropout 0.3, Dense 4 + softmax), Adam optimizer
2. **Fine-tuning:** last 4 layers of the backbone unfrozen, Adam with learning rate 1e-5

Loss: categorical cross-entropy. Batch size: 32.

| Model | Epochs (feature extraction + fine-tuning) | Approx. time per epoch |
| ----- | ----------------------------------------- | ---------------------- |
| VGG19 | 2 + 2 | about 76-82 s |
| DenseNet121 | 5 + 5 | about 64-70 s |
| MobileNetV2 | 5 + 5 | about 63 s |

## Results

Evaluated on the 4,233 held-out validation images.

| Model | Accuracy | Precision | Recall | F1-score | AUC |
| ----- | -------- | --------- | ------ | -------- | --- |
| VGG19 | 0.897 | 0.898 | 0.897 | 0.896 | 0.983 |
| DenseNet121 | 0.873 | 0.873 | 0.873 | 0.872 | 0.973 |
| **MobileNetV2** | **0.906** | **0.906** | **0.906** | **0.906** | **0.983** |

Per-class F1-score:

| Class | VGG19 | DenseNet121 | MobileNetV2 |
| ----- | ----- | ----------- | ----------- |
| COVID | 0.92 | 0.84 | 0.92 |
| Lung Opacity | 0.85 | 0.83 | 0.87 |
| Normal | 0.91 | 0.90 | 0.92 |
| Viral Pneumonia | 0.91 | 0.94 | 0.94 |

**Key findings**

- MobileNetV2 gives the best overall results (90.6% accuracy, AUC 0.983) while being by far the lightest model.
- VGG19 reaches almost the same accuracy (89.7%) with only 2 + 2 epochs, but it is the heaviest model.
- DenseNet121 is the weakest here (87.3%), mainly on COVID (recall 0.82) and Lung Opacity (recall 0.80).
- Lung Opacity is the hardest class for all models (recall between 0.80 and 0.85), it is often confused with Normal.
- Viral Pneumonia is well detected by all models (F1 of 0.91 to 0.94) despite being the smallest class.

## Main Takeaways

1. A lightweight model (MobileNetV2) is enough: more parameters do not mean better results on this task.
2. Transfer learning with only the last 4 layers fine-tuned already gives about 90% accuracy on 4 classes.
3. Looking at per-class recall matters more than global accuracy, especially for the COVID class.
4. Class overlap (Lung Opacity vs Normal) limits performance more than the choice of architecture.

## Limitations

- The same split is used for validation during training and for the final evaluation, so there is no fully independent test set. Reported scores may be slightly optimistic.
- The comparison is not fully equal: VGG19 was trained for 2 + 2 epochs vs 5 + 5 for the others, and MobileNetV2 uses its own preprocessing.
- No data augmentation and no class balancing were used.
- The split is done at image level; if several images come from the same patient, data leakage is possible.
- The models were not validated clinically and must not be used for real diagnosis.

## Repository Structure

```
notebooks/
├── vgg19_covid.ipynb          VGG19 training and evaluation
├── densenet121_covid.ipynb    DenseNet121 training and evaluation
└── mobilenetv2_covid.ipynb    MobileNetV2 training, evaluation and model saving
README.md
```

## How to Run

1. Open the notebooks on Kaggle (recommended) or locally with a GPU.
2. Add the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) dataset to the notebook.
3. Install the dependencies:

```
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Next Steps

- Train all models with the same number of epochs and add data augmentation
- Create a separate test set (train / validation / test), ideally split by patient
- Handle class imbalance (class weights or oversampling)
- Add Grad-CAM visualizations to check that models look at the lungs
- Test ResNet50, EfficientNet and Vision Transformers

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis.

## Author

**errahymymohamed01**: [GitHub](https://github.com/errahymymohamed01)
