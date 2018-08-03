from __future__ import division
import config
import read_image
from pca import calculate_pca
import cv2
from matplotlib import pyplot
import numpy
import sys
from sklearn.decomposition import PCA

flattened_matrix = read_image.read_images_to_db()

pca = PCA(svd_solver='randomized')
pca.fit(flattened_matrix)

print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))

ratios = []
for i in range(len(pca.explained_variance_ratio_)):
	sum = 0
	for j in range(i):
		sum += pca.explained_variance_ratio_[j]

	ratios.append(sum / numpy.sum(pca.explained_variance_ratio_))

pyplot.plot(ratios)
pyplot.ylabel(range(len(pca.explained_variance_ratio_)))
pyplot.show()