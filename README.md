# Image-retrieval-based-EfficientNet-B0
Use the pre-trained model of the network, add a new fully connected layer, and train on your own data set. When performing feature extraction, the final classifier layer is removed, and 512-dimensional features of the image are obtained for retrieval.
## Usage
  * Install with `pip install efficientnet_pytorch`. For more information, please go to https://github.com/lukemelas/EfficientNet-PyTorch
  
  * In `main.py`,To modify related parameters, please change the path of the training set and test set to your own path. This file is mainly for classification training to obtain a model suitable for your own data set.
  
    * train
      *  BEAR
         *  BEAR_0.jpg
         *  BEAR_1.jpg
         *  ...
      * CATS
         * CATS_0.jpg
         * CATS_1.jpg
         * ...
      * ...
    * val
      *  BEAR
         *  BEAR_100.jpg
         *  BEAR_101.jpg
         *  ...
      * CATS
         * CATS_100.jpg
         * CATS_101.jpg
         * ...
      * ...

  * `extract_feats.py`. Modify your own database path (database to be retrieved), and then run the program, the characteristics of all pictures in the database will be stored in a dictionary for later retrieval.
    * database
       * 0.jpg
       * 1.jpg
       * 2.jpg
       * 3.jpg
       * ... 
  
  * `retrieval.py`. Enter the path of the picture you want to retrieve, then you will get the five images that the program thinks are most similar.



  
  
  
