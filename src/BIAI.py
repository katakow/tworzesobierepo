# %% imports
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
from som import SOM, mse


# %% argumenty
block_width = 4
block_height = 4
bits_per_codevector = 8
vector_dimension = block_width * block_height
codebook_size = pow(2, bits_per_codevector)

epochs = 3

initial_learning_rate = 0.7

# %% source image
image_location = "C:\\git\\tworzesobierepo\\data\\kwadrat.png"
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)


print(type(image))

image_height = len(image)
image_width = len(image[0])
image_height
image_width

# %% resize
max_height = int(int(image_height / block_height) * block_height)
max_width = int(int(image_width / block_width) * block_width)

image = image[:max_height, :max_width]
image_height = len(image)
image_width = len(image[0])
image_height
image_width

# %% dividing the image into 4*4 blocks of pixels
image_vectors = []
for i in range(0, image_height, block_height):
    for j in range(0, image_width, block_width):
        image_vectors.append(
            np.reshape(
                image[i : i + block_width, j : j + block_height], vector_dimension
            )
        )
image_vectors = np.asarray(image_vectors).astype(float)
number_of_image_vectors = image_vectors.shape[0]

# %% properties of the SOM grid
som_rows = int(pow(2, int((log(codebook_size, 2)) / 2)))
som_columns = int(codebook_size / som_rows)

som = SOM(
    som_rows,
    som_columns,
    vector_dimension,
    epochs,
    number_of_image_vectors,
    initial_learning_rate,
    max(som_rows, som_columns) / 2,
)
reconstruction_values = som.train(image_vectors)

image_vector_indices, distance = vq(image_vectors, reconstruction_values)

image_after_compression = np.zeros([image_width, image_height], dtype="uint8")
for index, image_vector in enumerate(image_vectors):
    start_row = int(index / (image_width / block_width)) * block_height
    end_row = start_row + block_height
    start_column = (index * block_width) % image_width
    end_column = start_column + block_width
    image_after_compression[start_row:end_row, start_column:end_column] = np.reshape(
        reconstruction_values[image_vector_indices[index]], (block_width, block_height)
    )

output_image_name = "CB_size=" + str(codebook_size) + ".png"
# %% scipy.misc.imsave(output_image_name, image_after_compression)
imageio.imsave(output_image_name, image_after_compression)
print("Mean Square Error = ", mse(image, image_after_compression))

# %%
