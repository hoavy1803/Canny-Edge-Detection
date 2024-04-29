import numpy as np
import canny 
import math

convolution_dict = {}


def convolution_worker(pad_arr, pad_arr_out, arr_shape, kernel, kernel_shape, index):
    convolution_dict["pad_arr"] = pad_arr
    convolution_dict["pad_arr_out"] = pad_arr_out
    convolution_dict["arr_shape"] = arr_shape
    convolution_dict["kernel"] = kernel
    convolution_dict["kernel_shape"] = kernel_shape
    convolution_dict["index"] = index


def convolution_RGB_threaded(dimension):
    """
    Convolution threaded from every colour
    :param dimension: dimension of colour
    :return: array after convolution
    """

    pad_arr = np.frombuffer(convolution_dict["pad_arr"]).reshape(
        convolution_dict["arr_shape"]
    )
    pad_arr_out = np.frombuffer(convolution_dict["pad_arr_out"]).reshape(
        convolution_dict["arr_shape"]
    )
    kernel = np.frombuffer(convolution_dict["kernel"]).reshape(
        convolution_dict["kernel_shape"]
    )
    index = convolution_dict["index"]
    x = pad_arr.shape[0]
    y = pad_arr.shape[1]
    m = kernel.shape[0]
    n = kernel.shape[1]

    for x_index in range(index, x - index):
        for y_index in range(index, y - index):
            pad_arr_out[x_index][y_index][dimension] = np.sum(
                np.multiply(
                    kernel,
                    pad_arr[
                        x_index - index : x_index + index + 1,
                        y_index - index : y_index + index + 1,
                        dimension,
                    ],
                )
            )
    return pad_arr_out


sobel_dict = {}


def sobel_worker(arr, arr_shape):
    sobel_dict["arr"] = arr
    sobel_dict["arr_shape"] = arr_shape


def sobel_threaded(option):
    arr = np.frombuffer(sobel_dict["arr"]).reshape(sobel_dict["arr_shape"])
    if option == 0:
        return ["x", canny.sobel_x(arr)]
    else:
        return ["y", canny.sobel_y(arr)]


magni_direc_dict = {}


def magni_direc_worker(magni, direc, dx, dy, diff, arr_shape):
    magni_direc_dict["magni"] = magni
    magni_direc_dict["direc"] = direc
    magni_direc_dict["dx"] = dx
    magni_direc_dict["dy"] = dy
    magni_direc_dict["diff"] = diff
    magni_direc_dict["arr_shape"] = arr_shape


def magni_direc_threaded(i):
    magni = np.frombuffer(magni_direc_dict["magni"]).reshape(
        magni_direc_dict["arr_shape"]
    )
    direc = np.frombuffer(magni_direc_dict["direc"]).reshape(
        magni_direc_dict["arr_shape"]
    )
    dx = np.frombuffer(magni_direc_dict["dx"]).reshape(magni_direc_dict["arr_shape"])
    dy = np.frombuffer(magni_direc_dict["dy"]).reshape(magni_direc_dict["arr_shape"])
    diff = magni_direc_dict["diff"]
    arr_shape = magni_direc_dict["arr_shape"]

    start = i
    if (i + diff) >= magni.shape[0]:
        end = magni.shape[0] - 1
    else:
        end = i + diff - 1

    for i in range(start, end + 1):
        for j in range(magni.shape[1]):
            magni[i][j] = math.sqrt((dx[i][j] * dx[i][j]) + (dy[i][j] * dy[i][j]))
            direc[i][j] = math.degrees(math.atan2(dy[i][j], dx[i][j]))


maxima_dict = {}


def maxima_worker(maxima, magni, direc, diff, arr_shape):
    maxima_dict["maxima"] = maxima
    maxima_dict["magni"] = magni
    maxima_dict["direc"] = direc
    maxima_dict["diff"] = diff
    maxima_dict["arr_shape"] = arr_shape


def maxima_threaded(i):
    maxima = np.frombuffer(maxima_dict["maxima"]).reshape(maxima_dict["arr_shape"])
    arr_mag = np.frombuffer(maxima_dict["magni"]).reshape(maxima_dict["arr_shape"])
    arr_dir = np.frombuffer(maxima_dict["direc"]).reshape(maxima_dict["arr_shape"])
    diff = maxima_dict["diff"]
    arr_shape = maxima_dict["arr_shape"]

    start = i
    if (i + diff) >= arr_mag.shape[0]:
        end = arr_mag.shape[0] - 2
    else:
        end = i + diff - 1

    for i in range(start, end + 1):
        for j in range(1, arr_mag.shape[1] - 1):
            value = arr_mag[i][j]
            angle = arr_dir[i][j]

            # Left and Right
            if (
                ((angle >= -22.5) and (angle <= 22.5))
                or (angle <= -157.5)
                or (angle >= 157.5)
            ):
                v1 = arr_mag[i][j + 1]
                v2 = arr_mag[i][j - 1]
            # Top Left and Bottom Right
            elif ((angle <= -112.5) and (angle >= -157.5)) or (
                (angle >= 22.5) and (angle <= 67.5)
            ):
                v1 = arr_mag[i + 1][j + 1]
                v2 = arr_mag[i - 1][j - 1]
            # Up and Down
            elif ((angle <= -67.5) and (angle >= -112.5)) or (
                (angle >= 67.5) and (angle <= 112.5)
            ):
                v1 = arr_mag[i + 1][j]
                v2 = arr_mag[i - 1][j]
            # Bottom Left and Top Right
            elif ((angle >= 112.5) and (angle <= 157.5)) or (
                (angle <= -22.5) and (angle >= -67.5)
            ):
                v1 = arr_mag[i + 1][j - 1]
                v2 = arr_mag[i - 1][j + 1]

            if canny.compare_gradient(value, v1, v2):
                maxima[i][j] = value


hysterisis_dict = {}


def hysterisis_worker(arr, edges, arr_shape, min_th, max_th, diff):
    hysterisis_dict["arr"] = arr
    hysterisis_dict["edges"] = edges
    hysterisis_dict["arr_shape"] = arr_shape
    hysterisis_dict["min_th"] = min_th
    hysterisis_dict["max_th"] = max_th
    hysterisis_dict["diff"] = diff


def hysteresis_threaded(i):
    """
    Hysteresis_thresholding code for every thread
    :param arr: numpy.array(float), input non-maximum suppression image
    :param edges: numpy.array(float)
    :param start: start index
    :param end: end index
    :param min_th: minimum threshold
    :param max_th: maximum threshold
    """
    edges = np.frombuffer(hysterisis_dict["edges"]).reshape(
        hysterisis_dict["arr_shape"]
    )
    arr = np.frombuffer(hysterisis_dict["arr"]).reshape(hysterisis_dict["arr_shape"])
    min_th = hysterisis_dict["min_th"]
    max_th = hysterisis_dict["max_th"]
    diff = hysterisis_dict["diff"]

    start = i
    if (i + diff) >= arr.shape[0]:
        start = 0
        end = arr.shape[0]
    else:
        end = i + diff - 1

    for i in range(start, end):
        for j in range(0, arr.shape[1]):
            if arr[i][j] >= max_th:
                edges[i][j] = 255
            elif arr[i][j] < min_th:
                edges[i][j] = 0

    changes = 2
    while changes > 0:
        # If find only one change in the previous run,
        # the next change, if occurs, will be around that pixel
        # Thus, instead of starting from 0 indexes,
        # Go one row and one column before and after the pixel of the last change
        if changes == 1:
            start_x = c_p[0] - 1
            end_x = c_p[0] + 1
            start_y = c_p[1] - 1
            end_y = c_p[1] + 1
        else:
            start_x = start + 1
            end_x = end - 1
            start_y = 1
            end_y = arr.shape[1] - 1
        changes = 0
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if (
                    (arr[i][j] >= min_th)
                    and (arr[i][j] < max_th)
                    and (edges[i][j] == 0)
                ):
                    if np.sum(edges[i - 1 : i + 2, j - 1 : j + 2] > 0):
                        edges[i][j] = 255
                        changes += 1
                        c_p = [i, j]
