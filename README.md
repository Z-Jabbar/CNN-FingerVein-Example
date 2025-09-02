# Finger Vein Recognition CNN (Python / TensorFlow)

This repository contains a **Python snippet** implementing a **Convolutional Neural Network (CNN)** for **finger vein recognition**.

## Notes
- This code is **only a part of a full CNN project** for finger vein recognition.
- Preprocessed training and test datasets are required (`train_data` and `test_data` directories).
- The following variables should be defined before running:
  - `img_height`, `img_width` : input image dimensions
  - `batch_size` : batch size for training/testing
  - `train_datagen`, `test_datagen` : ImageDataGenerator instances
  - `y_true`, `y_pred` : true and predicted labels for evaluation

## File
- `finger_vein_cnn_snippet.py` : Python snippet of CNN model, compilation, training generator setup, and evaluation metrics.

## How to Use
1. Ensure Python 3.x and TensorFlow are installed.
2. Prepare your finger vein dataset in the following structure:
/train_data
/class1
/class2
/test_data
/class1
/class2
3. Adjust the parameters (`img_height`, `img_width`, `batch_size`) and define the data generators.
4. Run the script
5. The script will print accuracy, precision, and recall. 


