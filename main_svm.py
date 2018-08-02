from __future__ import division
import config
import read_image
from pca import calculate_pca
import cv2
from matplotlib import pyplot
import numpy
import sys

(training_set, training_label, test_set, test_label) = read_image.create_cross_validation_sets()
distinct_classes = list(set(training_label))
mean_face = numpy.mean(training_set, axis = 0)

mean, eigenvectors = cv2.PCACompute(training_set, mean=None, maxComponents = config.NO_OF_PCA_COMPONENTS)

# CALCULATE MEAN FACE FOR EACH CLASS
all_mean_class_faces = {}
mean_class_faces = {}
training_set_size_ratio = (config.K_CROSS_VALIDATION - 1) / config.K_CROSS_VALIDATION
size_of_each_class = (int)(training_set_size_ratio * config.DEFAULT_NO_OF_FILES)
for i in range(config.DEFAULT_NO_OF_CLASSES):
	all_mean_class_faces[distinct_classes[i]] = numpy.zeros(shape=(size_of_each_class, config.DEFAULT_FLATTENED_SIZE))
	mean_class_faces[distinct_classes[i]] = numpy.zeros(shape=config.DEFAULT_FLATTENED_SIZE)

for i in range(len(training_label)):
	all_mean_class_faces[training_label[i]][i % size_of_each_class] = training_set[i]

for i in range(config.DEFAULT_NO_OF_CLASSES):
	mean_class_faces[distinct_classes[i]] = numpy.mean(all_mean_class_faces[distinct_classes[i]], axis = 0)

# CALCULATE THE OMEGA TO REPRESENT EACH FACE CLASS
omega_of_classes = {}
for i in range(config.DEFAULT_NO_OF_CLASSES):
	omega_of_classes[distinct_classes[i]] = numpy.zeros(shape=config.NO_OF_PCA_COMPONENTS)

for j in range(config.DEFAULT_NO_OF_CLASSES):
	mean_class_difference = mean_class_faces[distinct_classes[j]] - mean_face
	vector_of_weights = numpy.zeros(shape=config.NO_OF_PCA_COMPONENTS)
	for i in range(config.NO_OF_PCA_COMPONENTS):
		sum = 0
		for k in range(config.DEFAULT_FLATTENED_SIZE):
			sum += mean_class_difference[k] * eigenvectors[i][k] 
		vector_of_weights[i] = sum
	omega_of_classes[distinct_classes[j]] = vector_of_weights

# CONSTRUCT C1 AND C2

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

# PREDICT TEST SET
predictions = []

for k in range(len(test_label)):
	distance_to_classes = numpy.zeros(shape = config.DEFAULT_NO_OF_CLASSES)
	for i in range(config.DEFAULT_NO_OF_CLASSES):
		sum = 0
		for j in range(config.NO_OF_PCA_COMPONENTS):
			sum += (projector_of_test[k][j] - omega_of_classes[distinct_classes[i]][j])**2
		distance_to_classes[i] = sum

	min_distance = sys.maxsize + 1
	min = -1
	for i in range(config.DEFAULT_NO_OF_CLASSES):
		if min_distance > distance_to_classes[i]:
			min = i
			min_distance = distance_to_classes[i]

	predictions.append(distinct_classes[min])
	print("predict: " + distinct_classes[min] + ", actual: " + test_label[k])

count = 0
for i in range(len(test_label)):
	if test_label[i] == predictions[i]:
		count = count + 1

print("rate: " + str(count / len(test_label)))