# landsatCNN
This Repo. contains code used for data collection, cleaning as well as model codes for running CNNs on Landsat Images at village level
Description of Files:  
**decidingDimensionsAndClusteringVillages.ipynb**: This is used to run the base model and trying to decide Village Input Dimensions of CNN.  
**downloadLandsat.js**: This is run on Google Earth Engine to extract images of states as a whole. It automatically gets split during downloading.  
**download.py**: This was used to quickly download the extracted images on Google Drive to Azure VM very quickly.  
**extractTiff.py**: This is used to clip the villages from the state images. It uses raster.io package.  
**inceptionPretrain.py**: This is used to transfer learn followed by fine-tuning for 2 epochs each.  
**cnnMfromScratch.py**: This is used to train a neural network from scratch using CNN-M architechture with BN added after each layer.  
