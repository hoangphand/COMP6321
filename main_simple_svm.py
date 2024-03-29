from __future__ import division
import config
import read_image
from pca import calculate_pca
import cv2
from matplotlib import pyplot
import numpy
import sys
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

(training_set, training_label, test_set, test_label) = read_image.create_cross_validation_sets()

for i in range(len(training_label)):
	training_label[i] = int(training_label[i].replace("s", ""))
for i in range(len(test_label)):
	test_label[i] = int(test_label[i].replace("s", ""))

distinct_classes = list(set(training_label))
mean_face = numpy.mean(training_set, axis = 0)

mean, eigenvectors = cv2.PCACompute(training_set, mean=None, maxComponents = config.NO_OF_PCA_COMPONENTS)

# PROJECT TRAINING SET ON THE VECTOR OF WEIGHTS
projector_of_training= numpy.zeros(shape = (len(training_label), config.NO_OF_PCA_COMPONENTS))
mean_class_difference_training = []

for i in range(len(training_label)):
	mean_class_difference_training.append(training_set[i] - mean_face)

for i in range(len(training_label)):
	tmp_projector_of_training = numpy.zeros(shape = config.NO_OF_PCA_COMPONENTS)
	for j in range(config.NO_OF_PCA_COMPONENTS):
		sum = 0
		for k in range(config.DEFAULT_FLATTENED_SIZE):
			sum += mean_class_difference_training[i][k] * eigenvectors[j][k] 
		tmp_projector_of_training[j] = sum
	projector_of_training[i] = tmp_projector_of_training

# PROJECT TEST SET ON THE VECTOR OF WEIGHTS
projector_of_test = numpy.zeros(shape = (len(test_label), config.NO_OF_PCA_COMPONENTS))
mean_class_difference_test = []

for i in range(len(test_label)):
	mean_class_difference_test.append(test_set[i] - mean_face)

for i in range(len(test_label)):
	tmp_projector_of_test = numpy.zeros(shape = config.NO_OF_PCA_COMPONENTS)
	for j in range(config.NO_OF_PCA_COMPONENTS):
		sum = 0
		for k in range(config.DEFAULT_FLATTENED_SIZE):
			sum += mean_class_difference_test[i][k] * eigenvectors[j][k] 
		tmp_projector_of_test[j] = sum
	projector_of_test[i] = tmp_projector_of_test

# TRAINING SVM
c = 1000
gamma = 1e-09
rbf_svc = SVC(kernel = 'rbf', C = c, gamma=gamma)
rbf_svc.fit(projector_of_training, training_label)

predictions = rbf_svc.predict(projector_of_test)
count = 0
for i in range(len(test_label)):
	if test_label[i] == predictions[i]:
		count = count + 1

print("rate: " + str(count / len(test_label)))