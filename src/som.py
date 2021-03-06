# Image Compression using SOM

import numpy as np
from scipy import misc, spatial
from math import log, exp, pow
from sklearn.metrics import mean_squared_error
import cv2
from math import sqrt
from scipy.cluster.vq import vq
import scipy.misc
import imageio
import sys


def mse(image_a, image_b):
    # calculate mean square error between two images
    err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err


class SOM(object):
    def __init__(
        self, rows, columns, dimensions, epochs, number_of_input_vectors, alpha, sigma
    ):

        self.rows = rows
        self.columns = columns
        self.dimensions = dimensions
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.number_of_input_vectors = number_of_input_vectors
        self.number_of_iterations = self.epochs * self.number_of_input_vectors

        self.weight_vectors = np.random.uniform(
            0, 255, (self.rows * self.columns, self.dimensions)
        )

    def get_bmu_location(self, input_vector, weights):

        tree = spatial.KDTree(weights)
        bmu_index = tree.query(input_vector)[1]
        return np.array([int(bmu_index / self.columns), bmu_index % self.columns])

    def update_weights(self, iter_no, bmu_location, input_data):

        learning_rate_op = 1 - (iter_no / float(self.number_of_iterations))
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        distance_from_bmu = []
        for x in range(self.rows):
            for y in range(self.columns):
                distance_from_bmu = np.append(
                    distance_from_bmu, np.linalg.norm(bmu_location - np.array([x, y]))
                )

        neighbourhood_function = [
            exp(-0.5 * pow(val, 2) / float(pow(sigma_op, 2)))
            for val in distance_from_bmu
        ]

        final_learning_rate = [alpha_op * val for val in neighbourhood_function]

        for l in range(self.rows * self.columns):
            weight_delta = [
                val * final_learning_rate[l]
                for val in (input_data - self.weight_vectors[l])
            ]
            updated_weight = self.weight_vectors[l] + np.array(weight_delta)
            self.weight_vectors[l] = updated_weight

    def train(self, input_data):

        iter_no = 0
        for epoch_number in range(self.epochs):
            for index, input_vector in enumerate(input_data):
                bmu_location = self.get_bmu_location(input_vector, self.weight_vectors)
                self.update_weights(iter_no, bmu_location, input_vector)
                iter_no += 1
        return self.weight_vectors
