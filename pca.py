import numpy

def calculate_pca(matrix):
	(rows, cols) = matrix.shape

	for index in range(rows):
		mean = numpy.sum(matrix[index]) / cols

		for j in range(cols):
			matrix[index, j] = matrix[index, j] - mean

	return matrix