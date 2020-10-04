# Semantic-Segmentation-Label-Metrics

## Problem:
Most people in the deep learning and computer vision communities understand what image classification is: we want our model to tell us what single object or scene is present in the image. Classification is very coarse and high-level.

Many are also familiar with object detection, where we try to locate and classify multiple objects within the image, by drawing bounding boxes around them and then classifying what’s in the box. Detection is mid-level, where we have some pretty useful and detailed information, but it’s still a bit rough since we’re only drawing bounding boxes and don’t really get an accurate idea of object shape [1].

## Task:
The task is to compute a confusion matrix and classification report for each of the 10 image ids, and also the confusion matrix and classification report for the entire combined 10 image set. The `images/` file is the original RGB camera image that was labeled. The `labels/` and `inferences/` files contain the ground truth labels and model inferences,
respectively, and both are the same format. Each contains a monochromatic PNG image, where the [0,255] value of each pixel represents the class assigned to that pixel. An example of `images/`,`labels/` and `inferences/` is as following:



The confusion matrix is defined such that matrix[i][j] contains the number of times when a ground truth label of `i` is predicted as `j` in the inferences. Thus, a perfectly performing model’s confusion matrix will only have values along the diagonal (true positives), and every non-diagonal value corresponds to errors (false positives or false negatives). The classification report is a dictionary that has, for each classification, that class’s precision, recall, and f1_score. These have the following definitions:

precision = TruePositives / (TruePositives + FalsePositives)

recall = TruePositives / (TruePositives + FalseNegatives)

f1_score = 2.0 * ((precision * recall) / (precision + recall))


## How to run:
The only thing you have to do to get started is run the `Semantic_Segmentation_Label_Metrics.py`:

`python Semantic_Segmentation_Label_Metrics.py`


## References:
1. Semantic Segmentation with Deep Learning
https://towardsdatascience.com/semantic-segmentation-with-deep-learning-a-guide-and-code-e52fc8958823
