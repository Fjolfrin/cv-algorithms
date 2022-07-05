from cgi import print_form
from statistics import median
import numpy as np
import cv2
from convolution import conv1d, conv2d


# Filters........................................

# Mean filters.
mean_3x3 = (1 / 9) * np.ones(9).reshape(3, 3)
mean_4x4 = (1 / 16) * np.ones(16).reshape(4, 4)
mean_10x10 = (1 / 100) * np.ones(100).reshape(10, 10)

# Edge filters.
edge_sobel_vertical = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
edge_sobel_horizontal = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]])
edge_roberts_up = np.array([[0, -1],
                            [1, 0]])
edge_roberts_down = np.array([[-1, 0],
                              [0, 1]])


def add_noise(image:np.array, noise_type:str):
    """
       This function adds noise to an image. This noise can be either
       median noise or salt and pepper noise.
    Args:
        image (np.array): The input image.
        noise_type (str): the type of noise to be applied (mean / saltandpepper).

    Returns:
        np.array: the noisy image.
    """
    if noise_type == 'gaussian':  # In case the noise is gaussian.
        noise = np.random.normal(0, 10, image.shape)  # Generate a mask with values from a normal distribution.
        image = image.astype('float64') + noise  # Corrupt the image with the noise.
        image = np.clip(image, 0., 255.)  # Clipping the image values so they do not exceed standard 8-bit limits.

    if noise_type == 'saltandpepper':  # In case the noise is salt & pepper.
        # Construct a mask, the size of the input image, whose map's values are either 0, 1, or 2, with
        # probabilities 90%, 5%, and 5% respectively. Wherever the value is 0, the image cell remains as is.
        # If the value is 1, then we have pepper (meaning cell_value = 0), and otherwise
        # we have salt (cell_value = 255).
        case = np.random.choice((0, 1, 2), image.shape, p=[.9, .05, .05])
        image[case == 1] = 0.
        image[case == 2] = 255.
    
    return image


def median_filter(image:np.array, window_size:int):
    """
        This method applies a median filter of a defined size to an image.
    Args:
        image (np.array): The input image.
        window_size (int): The size of the window for the median filter.

    Returns:
        np.array: The filtered image.
    """

    # Making a padded version of the input image for a median filter.
    image_padded = np.zeros((image.shape[0] + window_size, image.shape[1] + window_size))
    image_padded[window_size // 2: image_padded.shape[0] - window_size // 2, 
              window_size // 2: image_padded.shape[1] - window_size // 2] = image
    output = np.zeros((image.shape[0] + 2, image.shape[1] + 2))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Sort all the elements of the active window.
            temp = np.sort(np.concatenate(image_padded[i:i+window_size, j:j+window_size]))
            output[i][j] = temp[4]  # Select the middle one and pass it to the output image.

    # Cut all padding from the filtered image to get the same dimensions as the 
    # input image.
    output = output[window_size // 2: output.shape[0] - window_size // 2,
                    window_size // 2: output.shape[1] - window_size // 2]
                

    return output


