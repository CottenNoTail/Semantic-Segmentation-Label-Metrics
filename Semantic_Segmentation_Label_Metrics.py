import numpy as np
import matplotlib.pyplot as plt
import requests
import json

image_ids = [
    '000000_10',
    '000001_10',
    '000002_10',
    '000003_10',
    '000004_10',
    '000005_10',
    '000006_10',
    '000007_10',
    '000008_10',
    '000009_10'
]

label_classes = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]

# Download label and inference file from the server. Iterate through the image_id:
for image_id in image_ids:
    r_label = requests.get(
        'https://storage.googleapis.com/aquarium-public/interview/semseg_metrics_coding_challenge/data/labels/' + image_id + '.png')
    with open('label_' + image_id + '.png', 'wb') as f_label:
        f_label.write(r_label.content)

    r_inference = requests.get(
        'https://storage.googleapis.com/aquarium-public/interview/semseg_metrics_coding_challenge/data/inferences/' + image_id + '.png')
    with open('inference_' + image_id + '.png', 'wb') as f_inference:
        f_inference.write(r_inference.content)

# Define a class name process to calculate matrix and report
class process(object):

    def __init__(self, label, inference):
        # Init the label and inference
        self.label = label
        self.inference = inference

        # Init an empty matrix and dict for storing the results
        self.matrix = np.zeros((19, 19))
        self.report = {}

        # Get the dimension of label
        self.dim_y = self.label.shape[1]
        self.dim_x = self.label.shape[0]

    # A function to calculate the Matrix
    def CalMatrix(self):

        """ Loop through each pixel in the label array and inference array.
        The pixel at position (i,j) in label array and inference means this
        pixel is classified as category inference[i][j] while its actual
        label should be label[i][j] """

        for i in range(self.dim_x):
            for j in range(self.dim_y):
                pix_label = int(self.label[i][j] * 255)
                pix_inference = int(self.inference[i][j] * 255)

                # Make sure the pixel value is within [0,18]
                if 0 <= pix_inference < 19 and 0 <= pix_label < 19:
                    self.matrix[pix_label][pix_inference] += 1

        return self.matrix

    def CalReport(self):
        for i in range(19):
            # TruePositives are the diagonal elements
            TruePositives = self.matrix[i][i]

            # FalsePositives are the row sum minus the ith element
            FalsePositives = np.sum(self.matrix, axis=0)[i] - self.matrix[i][i]

            # FalsePositives are the col sum minus the ith element
            FalseNegatives = np.sum(self.matrix, axis=1)[i] - self.matrix[i][i]

            # By definition
            precision = TruePositives / (TruePositives + FalsePositives) if TruePositives + FalsePositives != 0 else 0
            recall = TruePositives / (TruePositives + FalseNegatives) if TruePositives + FalseNegatives != 0 else 0
            f1_score = 2.0 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

            # Append f1_score, precision and recall to the report dict
            self.report[label_classes[i]] = {"f1_score": f1_score, "precision": precision, "recall": recall }

        return self.report




def main():
    # Loop through all the 10 images
    for image_id in image_ids:
        label = plt.imread('label_' + image_id + '.png')
        inference = plt.imread('inference_' + image_id + '.png')

        # Init a process class for each image to calculate matrix and report
        processing = process(label,inference)

        matrix = processing.CalMatrix()
        report = processing.CalReport()

        # Save Report as json file
        out_file = open(image_id + "Report", 'w+')
        json.dump(report, out_file)

        # Save matrix as txt file
        np.savetxt(image_id + "Matrix", matrix, fmt="%d", delimiter=',')


if __name__ == '__main__':
    main()
  

