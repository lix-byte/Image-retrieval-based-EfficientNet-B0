# Image-retrieval-based-EfficientNet-B0
Use the pre-trained model of the network, add a new fully connected layer, and train on your own data set. When performing feature extraction, the final classifier layer is removed, and 512-dimensional features of the image are obtained for retrieval.
## Usage
  * Install with `pip install efficientnet_pytorch`. For more information, please go to https://github.com/lukemelas/EfficientNet-PyTorch
  * In `main.py`,To modify related parameters, please change the path of the training set and test set to your own path. This file is mainly for classification training to obtain a model suitable for your own data set.
