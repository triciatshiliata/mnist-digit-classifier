# Handwritten Digit Classifier (MNIST) with PyTorch

A deep neural network that recognizes handwritten digits (0–9) from the MNIST dataset, built and trained in PyTorch.

**Result: 97.6% accuracy on 10,000 test images the model never saw during training.**

## How it works

1. **Data pipeline.** The 60,000 MNIST training images are normalized and split into a **50,000-image training set** and a **10,000-image validation set**. A separate 10,000-image test set is held back for the final score. Data is batched in groups of 64 with `DataLoader`.
2. **Model.** A fully connected network with three hidden layers and dropout to reduce overfitting:

   ```
   Input (28×28 = 784 pixels)
     → Linear 784 → 256 + ReLU
     → Dropout (p = 0.3)
     → Linear 256 → 128 + ReLU
     → Linear 128 → 64  + ReLU
     → Linear 64  → 10  (one output per digit)
   ```

3. **Training.** 5 epochs with the Adam optimizer (learning rate 0.001) and cross-entropy loss. Validation loss and accuracy are tracked after every epoch to watch for overfitting.
4. **Hardware.** The code automatically uses a GPU (CUDA) when one is available and falls back to the CPU otherwise.

## Results

| Epoch | Train loss | Validation loss | Validation accuracy |
|---|---|---|---|
| 1 | 0.3225 | 0.1608 | 95.04% |
| 2 | 0.1512 | 0.1162 | 96.76% |
| 3 | 0.1169 | 0.0963 | 97.21% |
| 4 | 0.1011 | 0.0801 | 97.60% |
| 5 | 0.0872 | 0.0859 | 97.38% |

**Final test set:** loss 0.0826 · **accuracy 97.60%**

## Run it yourself

```bash
git clone https://github.com/triciatshiliata/mnist-digit-classifier.git
cd mnist-digit-classifier
pip install -r requirements.txt
jupyter notebook mnist_classifier.ipynb
```

The MNIST dataset downloads automatically into a `data/` folder the first time you run the notebook.

## Tech stack

Python · PyTorch · torchvision · Jupyter Notebook

## Possible next steps

- Swap the fully connected layers for a convolutional neural network (CNN), which usually gets above 99% on MNIST
- Add a confusion matrix to see which digits get mixed up most often
- Save the trained model and build a small app where you can draw a digit and get a prediction

## Author

**Patricia Tshiliata** · [LinkedIn](https://www.linkedin.com/in/patricia-tshiliata) · [GitHub](https://github.com/triciatshiliata)
