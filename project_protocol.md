# Project Protocol

## 1. Main Research Question

How do classical machine learning, neural networks and hybrid models compare for four-class brain tumour classification using MRI images under the same experimental conditions?


## 2. Project Aim

The aim of this project is to compare classical machine learning models, neural networks and hybrid models for four-class brain tumour classification using the same MRI dataset and experimental setup.

More specifically: 
- Classical models: SVM and Random Forest
- Neural networks: ResNet50 and EfficientNetB0
- Hybrid models: CNN feature extraction combined with SVM or Random Forest


## 3. Project Objectives

1. To review existing research on brain tumour classification and identify representative classical machine learning, neural networks and hybrid approaches.

2. To design and implement the selected models using consistent preprocessing, training procedures and experimental conditions.

3. To evaluate all models using appropriate classification metrics, including accuracy, precision, recall, F1-score, and confusion matrices. 

4. To compare the results and critically analyse the strengths, weaknesses and computational requirements of each approach.


## 4. Dataset and Classification Task

The project will use the Kaggle Brain Tumour MRI Dataset, containing labelled MRI images from four classes: glioma, meningioma, pituitary tumour and no tumour.

This is a supervised multi-class classification task in which each MRI image is assigned to one of the four tumour categories. 


## 5. Models to be compared

The following six models will be implemented and compared:

### Classical Machine Learning Models

- HOG and GLCM features with SVM
- HOG and GLCM features with Random Forest

### Neural Network Models

- ResNet50 using transfer learning
- EfficientNetB0 using transfer learning

### Hybrid Models

- ResNet50 feature extraction with SVM
- ResNet50 feature extraction with Random Forest


## 6. Evaluation Metrics

The models will be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For precision, recall and F1-score, both macro-averaged and per-class results will be reported. We use macro-average as it gives equal importance to each of the four classes, even if the dataset is imbalanced.

The confusion matrix will be used to identify which tumour classes are most frequently confused with one another.

Training time and inference time will also be recorded to compare computational cost for the different approaches.


## 7. Experimental Design








## 8. Data Preprocessing

Before modelling, the dataset will be inspected to identify unreadable images, incorrect class labels and duplicated images. The number of images in each class will also be recorded to determine whether class imbalance is present.

The original image dimensions will also be inspected. A consistent input resolution will then be selected based on the dataset characteristics, model requirements and computational availability. 

The preprocessing will also differ slightly according to the modelling approach:

### Classical Machine Learning Models
- Images will be converted to grayscale if not already in that format. 
- HOG features will be extracted to represent edges and shapes.
- GLCM features will be extracted to represent image texture.
- The HOG and GLCM features will be combined into one feature vector for each image. 
- Feature scaling for SVM will be fitted only using the training part of each fold and then applied to the corresponding validation part.

### Neural Network Models
- Images will be converted to three-channel RGB format because the pre-trained ResNet50 and EfficientNetB0 models expect three input channels.
- The model-specific preprocessing function provided by TensorFlow/Keras will be applied. 
- Data Augmentation will be applied only to the training images.
- Augmentation will consist of small rotations, translations and zoom transformations

### Hybrid Models:

- Images will undergo the same preprocessing required by the pre-trained ResNet50 model
- ResNet50 will be used to extract deep feature vectors from the images.
- The extracted features will then be used as input to the SVM and Random Forest classifiers.

Any preprocessing operation that requires learning parameters from the data will be applied only on the training data. This will prevent the information from the test data leaking into the model training. 



## 9. Statistical Comparison

The performance of the six models will be compared using the results obtained from the stratified cross-validation folds. Using the same folds will allow the model scores to be treated as paired observations.

Macro-averaged F1-score will be used as the primary measure for statistical comparison because it gives equal importance to all four tumour classes. Accuracy, precision, recall and other evaluation measures will also be reported as secondary results. 

The following procedure will be used:

### Comparing models within each category
- HOG and GLCM with SVM versus HOG and GLCM with Random Forest
- ResNet50 versus EfficientNetB0
- ResNet50 features with SVM versus ResNet50 features with Random Forest

### Comparing models between categories
- HOG and GLCM with SVM versus ResNet50 features with SVM
- HOG and GLCM with Random Forest versus ResNet50 features with Random Forest
- ResNet50 as an end-to-end neural network versus hybrid models using ResNet50 and SVM or Random Forest


Where appropriate, effect sizes will also be reported to indicate the magnitude of the performance differences rather than relying only on statistical significance.

The Wilcoxon signed-rank test will be used to compare paired performance scores obtained from the same
cross-validation folds.

Because the number of cross-validation folds is relatively small and the fold results are not completely independent, the statistical findings will be interpreted as supporting evidence rather than definitive proof that one model is universally better.

A significance level of 0.05 will be used. If several related statistical tests are conducted, the Holm correction will be applied to reduce the risk of false-positive results.


## 10. Reproducibility and Experiment Recording

All experiments will be documented so that the results can be reproduced and checked. 

The following information will be recorded:
- The Python version and the version of all the libraries used
- The random seeds used for the dataset splitting, model training and evaluation
- The exact stratified cross-validation fold assignment
- Image preprocessing and augmentation techniques
- Feature extraction settings for HOG, GLCM and ResNet50
- Model architecture and hyperparameter settings
- Training and validation performance for each fold
- Training time and inference time
- Final evaluation metrics, confusion matrices and statistical test results
- The hardware used such as Google Colab and the available GPU

The same fold assignment will be saved and reused across the model wherever computationally feasible. This will ensure that differences in performance are not caused by evaluating models on different groups of images.

The required Python packages and their versions will be recorded in a requirements.txt file. Code, configuration files and experiment results will be organised in the GitHub repository. Large datasets and trained model files will be uploaded directly to GitHub, but instructions for obtaining the dataset and for reproducing the experiments will be included in the project documentation.



