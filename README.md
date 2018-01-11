# Plant Seedlings Image Classification
This project is to built an image classifier for plant species by fine-tuning the top layers of a pre-trained CNN (Xception). To further improve the performance, a filter is created with OpenCV to mask the background of the images and improved the accuracy by 3%. Because the limit of the training data, an overfitting problem is observed and then solved by data augmentation and achieved 96% accuracy.

## Introduction to the Dataset
A training set and a test set of images of plant seedlings at various stages of grown. Each image has a filename that is its unique id. The dataset comprises 12 plant species. The list of species is as follows:

`Black-grass`
`Charlock`
`Cleavers`
`Common Chickweed`
`Common wheat`
`Fat Hen`
`Loose Silky-bent`
`Maize`
`Scentless Mayweed`
`Shepherds Purse`
`Small-flowered Cranesbill`
`Sugar beet`

The dataset is available here https://vision.eng.au.dk/plant-seedlings-dataset/ and https://www.kaggle.com/c/plant-seedlings-classification/data

## Libraries Used in This Project
* OpenCV

## Functions of Each File
* `Plant_Seedlings_EDA.ipynb` is the EDA of the dataset.
* `ps_load_data.py` loads the input data and generate pandas DataFrames contains the file paths, categories ids, categories, etc.
* `ps_image_to_array.py` process the training dataset without filtering the background.
* `ps_image_to_array_filter.py` process the training dataset and filter the background.
* `ps_image_to_array_filter_test.py` process the image data for the test dataset.
* `ps_fine_tune_part1.py` use the bottleneck features and train the dense layer.
* `ps_fine_tune_part2.py` re-train the top layers of the Xception model.
* `main.py` is the main function.

## Possible Further Improvements
* Consider other pre-trained model, e.g. InceptionResNetV2.
* Generate more training data.
* Increase the number of layers to be re-trained.

## References
Thanks to the authors below who provide excellent kernels:

https://www.kaggle.com/gaborvecsei/plant-seedlings-fun-with-computer-vision

https://www.kaggle.com/gaborfodor/seedlings-pretrained-keras-models
