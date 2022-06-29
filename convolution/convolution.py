import string
import numpy as np

def conv1d(a: np.array, b: np.array, padding: string='full', stride: int=1):
    """
    This method implements the single-dimensional linear convolution operation.

    Args:
        a (np.array): the input signal.
        b (np.array): the to-be-applied filter signal.
        padding (string): the type of padding to be applied (full, same, valid) [default='full'].
        stride (int): the step with which the filter moves 'scans' the input signal [default=1].
    """
    # Convert input signals to numpy arrays if they are given as lists.
    # If they are already numpy arrays, just pass them to ther variables.
    if type(a) != np.array:
        signal = np.array(a)
    else:
        signal = a
    
    if type(b) != np.array:
        filter = np.array(b)
    else:
        filter = b

    # Isolate the length of both input and filter signals.
    N = signal.shape[0]
    M = filter.shape[0]

    # Invert filter signal.
    filter_inv = filter[::-1]

    # Set the length of the output signal, and create a padded version
    # of the input signal for easier calculations.
    # Both of these two are dependent on the type of padding to be used.
    if padding == "full":
        out_len = np.ceil((N + M - 1) / stride, dtype=int)
        signal_padded = np.pad(signal, (M - 1, M - 1), mode='constant', constant_values=0)

    elif padding == "same":
        out_len = np.ceil(N / stride, dtype=int)
        signal_padded = np.pad(signal, (M // 2, M // 2), mode='constant', constant_values=0)

    elif padding == "valid":
        out_len = np.ceil((N - M + 1) / stride, dtype=int)
        signal_padded = signal
    
    # Create a zero-filled output array with the output length defined earlier.
    out = np.zeros(int(out_len))

    # Calculate the convolution operation result for every cell of the output array.
    for i in range(out.shape[0]):
        out[i] = np.sum(signal_padded[i * stride: i * stride + M] * filter_inv)
    
    return out


def conv2d(a: np.array, b: np.array, padding: string='full', stride: int=1):
    """
    This method implements the two-dimensional linear convolution operation.

    Args:
        a (np.array): the input array.
        b (np.array): the to-be-applied filter array.
        padding (string): the type of padding to be applied (full, same, valid) [default='full'].
        stride (int): the step with which the filter moves 'scans' the input array [default=1].
    """
    # Convert input signals to numpy arrays if they are given as lists.
    # If they are already numpy arrays, just pass them to ther variables.
    if type(a) != np.array:
        image = np.array(a)
    else:
        image = a
    
    if type(b) != np.array:
        filter = np.array(b)
    else:
        filter = b

    # Isolate the length of both input and filter arrays.
    N, M = image.shape
    P, Q = filter.shape

    # Invert filter array.
    filter_inv = filter[::-1, ::-1]

    # Set the length of the output array, and create a padded version
    # of the input image for easier calculations.
    # Both of these two are dependent on the type of padding to be used.
    if padding == "full":
        out_len = [np.ceil((N + P - 1) / stride), np.ceil((M + Q - 1) / stride)]
        image_padded = np.pad(image, ((P - 1, P - 1), (Q - 1, Q - 1)),
                              mode='constant', constant_values=0.)

    elif padding == "same":
        out_len = [np.ceil(N / stride), np.ceil(M / stride)]
        image_padded = np.pad(image, ((P // 2, P // 2), (Q // 2, Q // 2)), 
                              mode='constant', constant_values=0.)

    elif padding == "valid":
        out_len = [np.ceil((N - P + 1) / stride), np.ceil((M - Q + 1) / stride)]
        image_padded = image
    
    # Create a zero-filled output array with the output length defined earlier.
    out = np.zeros((int(out_len[0]), int(out_len[1])))

    # Calculate the convolution operation result for every cell of the output array.
    for i in range(out.shape[0]):  # Iterate through rows...
        for j in range(out.shape[1]):  # Iterate through collumns...

            out[i,j] = np.sum(image_padded[i * stride: i * stride + P, j * stride: j * stride + Q] * filter_inv)
    
    return out


def conv3d(a: np.array, b: np.array, padding: string='full', stride: int=1):
    """
    This method implements the three-dimensional linear convolution operation.

    Args:
        a (np.array): the input array.
        b (np.array): the to-be-applied filter array.
        padding (string): the type of padding to be applied (full, same, valid) [default='full'].
        stride (int): the step with which the filter moves 'scans' the input array [default=1].
    """
    # Convert input signals to numpy arrays if they are given as lists.
    # If they are already numpy arrays, just pass them to ther variables.
    if type(a) != np.array:
        image = np.array(a)
    else:
        image = a
    
    if type(b) != np.array:
        filter = np.array(b)
    else:
        filter = b

    # Isolate the length of both input and filter arrays.
    N, M, O = image.shape
    P, Q, R = filter.shape

    # Invert filter array.
    filter_inv = filter[::-1, ::-1, ::-1]

    # Set the length of the output array, and create a padded version
    # of the input image for easier calculations.
    # Both of these two are dependent on the type of padding to be used.
    if padding == "full":
        out_len = [np.ceil((N + P - 1) / stride), np.ceil((M + Q - 1) / stride), (O + R - 1) / stride]
        image_padded = np.pad(image, ((P - 1, P - 1), (Q - 1, Q - 1), (R - 1, R - 1)),
                              mode='constant', constant_values=0.)

    elif padding == "same":
        out_len = [np.ceil(N / stride), np.ceil(M / stride), np.ceil(O / stride)]
        image_padded = np.pad(image, ((P // 2, P // 2), (Q // 2, Q // 2), (R // 2, R // 2)), 
                              mode='constant', constant_values=0.)

    elif padding == "valid":
        out_len = [np.ceil((N - P + 1) / stride), np.ceil((M - Q + 1) / stride), np.ceil((O - R + 1) / stride)]
        image_padded = image
    
    # Create a zero-filled output array with the output length defined earlier.
    out = np.zeros((int(out_len[0]), int(out_len[1]), int(out_len[2])))

    # Calculate the convolution operation result for every cell of the output array.
    for i in range(out.shape[0]):  # Iterate height...
        for j in range(out.shape[1]):  # Iterate width...
            for k in range(out.shape[2]):  # Iterate depth...
                out[i, j, k] = np.sum(image_padded[i * stride: i * stride + P, 
                                                   j * stride: j * stride + Q,
                                                   k * stride: k * stride + R] * filter_inv)
    
    return out