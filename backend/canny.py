import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
from multiprocessing import Pool, RawArray
import multiprocess_helper as mph

# =========================================================
# If use_processes is set to True, the number of processes specified in num_of_processes will be used
# Convolution: 3 parallel processes in RGB, each for every colour
# Derivatives: 2 parallel processes, one for dx, one for dy
# Magnitude/Direction: num_of_processes parallel processes, divide array respectively for each process
# Non-Maximum-Suppression: num_of_processes parallel processes, divide array respectively for each process
# Hysterisis-Thresholding: num_of_processes parallel processes, divide array respectively for each process

# For best results use the number of your CPU physical cores
# num_of_processes = 6
# ==========================================================


# ===================HELPER FUNCTIONS=======================
def flipKernel(kernel):
    """
    Flip a kernel 180 degress
    :param kernel: input kernel
    :return: verti: flipped kernel
    """
    horiz = np.fliplr(kernel)
    verti = np.flipud(horiz)
    return verti


def deletePadding(arr, size):
    """
    Delete the padding added around the image
    :param arr: Image with padding
    :param size: Size of the padding
    :return: arr: Image without padding
    """
    return arr[size:-size, size:-size]


def normalizeInRange(arr, option):
    """
    Normalize a 2D numpy array
    :param arr: Array with values
    :param option: if 0, normalize 0 to 1, if -1 normalize -1 to 1
    """
    x, y = arr.shape
    max = np.max(arr)
    min = np.min(arr)

    if option == -1:
        for i in range(x):
            for j in range(y):
                arr[i][j] = 2 * ((arr[i][j] - min) / (max - min)) - 1
    else:
        for i in range(x):
            for j in range(y):
                arr[i][j] = (arr[i][j] - min) / (max - min)


def magnitude_and_direction(dx, dy):
    """
    Calculate magnitute and direction for gradient
    :param dx: Partial derivatives x
    :param dy: Partial derivatives y
    :return:
    magni: array with magnitude for every pixel
    direc array with direction for every pixel
    """
    use_processes = True
    num_of_processes = 8
    magni = np.zeros_like(dx, dtype=float)
    direc = np.zeros_like(dx, dtype=float)

    if use_processes:
        # Using processes
        magni_X = RawArray("d", magni.shape[0] * magni.shape[1])
        magni_X_np = np.frombuffer(magni_X).reshape(magni.shape)
        np.copyto(magni_X_np, magni)

        direc_X = RawArray("d", direc.shape[0] * direc.shape[1])
        direc_X_np = np.frombuffer(direc_X).reshape(direc.shape)
        np.copyto(direc_X_np, direc)

        dx_X = RawArray("d", dx.shape[0] * dx.shape[1])
        dx_X_np = np.frombuffer(dx_X).reshape(dx.shape)
        np.copyto(dx_X_np, dx)

        dy_X = RawArray("d", dy.shape[0] * dy.shape[1])
        dy_X_np = np.frombuffer(dy_X).reshape(dy.shape)
        np.copyto(dy_X_np, dy)

        diff = math.floor(magni.shape[0] / num_of_processes)

        with Pool(
            processes=num_of_processes,
            initializer=mph.magni_direc_worker,
            initargs=(magni_X, direc_X, dx_X, dy_X, diff, magni.shape),
        ) as pool:
            result = pool.map(mph.magni_direc_threaded, range(0, magni.shape[0], diff))

        magni = magni_X_np
        direc = direc_X_np
    else:
        for i in range(magni.shape[0]):
            for j in range(magni.shape[1]):
                magni[i][j] = math.sqrt((dx[i][j] * dx[i][j]) + (dy[i][j] * dy[i][j]))
                direc[i][j] = math.degrees(math.atan2(dy[i][j], dx[i][j]))

    normalizeInRange(magni, 0)
    return magni, direc


def compare_gradient(c, v1, v2):
    """
    Compare if value is greater than both v1 and v2
    :param c: middle value
    :param v1: first neighbour value
    :param v2: second neighbour value
    :return: True if grater, else False
    """
    if (c >= v1) and (c >= v2):
        return True
    else:
        return False


def sobel_threaded_manager(gau_img):
    dx = np.empty((gau_img.shape[0], gau_img.shape[1]), dtype=float)
    dy = np.empty((gau_img.shape[0], gau_img.shape[1]), dtype=float)

    gau_img_X = RawArray("d", gau_img.shape[0] * gau_img.shape[1])
    gau_img_X_np = np.frombuffer(gau_img_X).reshape(gau_img.shape)
    np.copyto(gau_img_X_np, gau_img)

    with Pool(
        processes=2, initializer=mph.sobel_worker, initargs=(gau_img_X, gau_img.shape)
    ) as pool:
        result = pool.map(mph.sobel_threaded, range(0, 2))

    if result[0][0] == "x":
        return result[0][1], result[1][1]
    else:
        return result[1][1], result[0][1]


# ===================HELPER FUNCTIONS=======================


# ====================MAIN FUNCTIONS========================
def convolution_2D(arr, kernel, border_type):
    """
    Calculate the 2D convolution arr*kernel
    :param arr: numpy.array(float), input array
    :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions allowed)
    :param border_type: int, padding method (OpenCV)
    :return:
    conv_arr: numpy.array(float), convolution output
    """

    n, m = kernel.shape
    assert n == m, "Kernel is not square!"
    assert n % 2 != 0, "Kernel has not odd size"

    flipped_kernel = flipKernel(kernel)
    pad_amount = math.floor(n / 2)
    pad_arr = cv2.copyMakeBorder(
        arr, pad_amount, pad_amount, pad_amount, pad_amount, border_type
    )

    # Check if RGB or Grayscale
    if pad_arr.ndim > 2:
        rgb = True
        x, y, s = pad_arr.shape
        pad_arr_out = np.zeros_like(pad_arr, dtype=float)
    else:
        rgb = False
        x, y = pad_arr.shape
        s = 1
        pad_arr_out = np.zeros_like(pad_arr, dtype=float)

    index = math.floor(n / 2)
    use_processes = True
    num_of_processes = 8
    if rgb:
        if use_processes:
            # Compute parallel the 3 colours to save some time
            # Using processes
            pad_arr_X = RawArray(
                "d", pad_arr.shape[0] * pad_arr.shape[1] * pad_arr.shape[2]
            )
            pad_arr_X_np = np.frombuffer(pad_arr_X).reshape(pad_arr.shape)
            np.copyto(pad_arr_X_np, pad_arr)

            pad_arr_out_X = RawArray(
                "d", pad_arr_out.shape[0] * pad_arr_out.shape[1] * pad_arr_out.shape[2]
            )
            pad_arr_out_X_np = np.frombuffer(pad_arr_out_X).reshape(pad_arr_out.shape)
            np.copyto(pad_arr_out_X_np, pad_arr_out)

            kernel_X = RawArray("d", kernel.shape[0] * kernel.shape[1])
            kernel_X_np = np.frombuffer(kernel_X).reshape(kernel.shape)
            np.copyto(kernel_X_np, flipped_kernel)

            with Pool(
                processes=3,
                initializer=mph.convolution_worker,
                initargs=(
                    pad_arr_X,
                    pad_arr_out_X,
                    pad_arr.shape,
                    kernel_X,
                    kernel.shape,
                    index,
                ),
            ) as pool:
                result = pool.map(mph.convolution_RGB_threaded, range(0, 3))

            pad_arr_out = pad_arr_out_X_np
        else:
            for d in range(0, s):
                # For every colour
                for x_index in range(index, x - index):
                    for y_index in range(index, y - index):
                        pad_arr_out[x_index][y_index][d] = np.sum(
                            np.multiply(
                                kernel,
                                pad_arr[
                                    x_index - index : x_index + index + 1,
                                    y_index - index : y_index + index + 1,
                                    d,
                                ],
                            )
                        )
    else:
        for x_index in range(index, x - index):
            for y_index in range(index, y - index):
                pad_arr_out[x_index][y_index] = np.sum(
                    np.multiply(
                        kernel,
                        pad_arr[
                            x_index - index : x_index + index + 1,
                            y_index - index : y_index + index + 1,
                        ],
                    )
                )

    conv_arr = deletePadding(pad_arr_out, pad_amount)
    return conv_arr


def gaussian_kernel_2D(ksize, sigma):
    """
    Calculate a 2D Gaussian kernel
    :param ksize: int, size of 2d kernel, always needs to be an odd number
    :param sigma: float, standard deviation of gaussian
    :return:
    kernel: numpy.array(float), ksize x ksize gaussian kernel with mean=0
    """
    assert ksize > 0, "Kernel size is zero!"
    assert ksize % 2 != 0, "Kernel has not odd size"
    assert sigma > 0, "Sigma is zero!"

    gau_kernel = np.zeros((ksize, ksize))
    center = math.floor(ksize / 2)
    for x in range(ksize):
        for y in range(ksize):
            exponent = -((x - center + 1) ** 2 + (y - center + 1) ** 2) / (
                2.0 * (sigma**2)
            )
            gau_kernel[x][y] = (1.0 / (2.0 * math.pi * sigma**2)) * pow(
                math.e, exponent
            )

    # Normalize kernel to sum to 1
    gau_kernel = gau_kernel / np.sum(gau_kernel)
    return gau_kernel


def sobel_x(arr):
    """
    Calculate 1st order partial derivatives along x-axis
    :param arr: numpy.array(float), input image
    :return:
    dx: numpy.array(float), output partial derivative
    """

    s_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    dx = convolution_2D(arr, s_x, cv2.BORDER_REPLICATE)
    normalizeInRange(dx, -1)
    return dx


def sobel_y(arr):
    """
    Calculate 1st order partial derivatives along y-axis
    :param arr: numpy.array(float), input image
    :return:
    dy: numpy.array(float), output partial derivatives
    """

    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    dy = convolution_2D(arr, s_y, cv2.BORDER_REPLICATE)
    normalizeInRange(dy, -1)
    return dy


def non_maximum_suppression(arr_mag, arr_dir):
    """
    Find local maxima along image gradient direction
    :param arr_mag: numpy.array(float), input image gradient magnitude
    :param arr_dir: numpy.array(float), input image gradient direction
    :return:
    arr_local_maxima: numpy.array(float)
    """
    v1, v2 = 0, 0
    arr_local_maxima = np.zeros_like(arr_mag)
    use_processes = True
    num_of_processes = 8
    if use_processes:
        # Using processes
        arr_local_maxima_X = RawArray(
            "d", arr_local_maxima.shape[0] * arr_local_maxima.shape[1]
        )
        arr_local_maxima_X_np = np.frombuffer(arr_local_maxima_X).reshape(
            arr_local_maxima.shape
        )
        np.copyto(arr_local_maxima_X_np, arr_local_maxima)

        arr_mag_X = RawArray("d", arr_mag.shape[0] * arr_mag.shape[1])
        arr_mag_X_np = np.frombuffer(arr_mag_X).reshape(arr_mag.shape)
        np.copyto(arr_mag_X_np, arr_mag)

        arr_dir_X = RawArray("d", arr_dir.shape[0] * arr_dir.shape[1])
        arr_dir_X_np = np.frombuffer(arr_dir_X).reshape(arr_dir.shape)
        np.copyto(arr_dir_X_np, arr_dir)

        diff = math.floor(arr_mag.shape[0] / num_of_processes)

        with Pool(
            processes=num_of_processes,
            initializer=mph.maxima_worker,
            initargs=(arr_local_maxima_X, arr_mag_X, arr_dir_X, diff, arr_mag.shape),
        ) as pool:
            result = pool.map(mph.maxima_threaded, range(1, arr_mag.shape[0], diff))

        arr_local_maxima = arr_local_maxima_X_np
    else:
        for i in range(1, arr_mag.shape[0] - 1):
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

                if compare_gradient(value, v1, v2):
                    arr_local_maxima[i][j] = value

    return arr_local_maxima


def hysteresis_thresholding(arr, low_ratio, high_ratio):
    """
    Use low and high ratio to threshold the non-maximum suppression image and then link non-weak edges
    :param arr: numpy.array(float), input non-maximum suppression image
    :param low_ratio: float, low threshold ratio
    :param high_ratio: float, high threshold ratio
    :return:
    edges: numpy.array(float)
    """

    edges = np.zeros_like(arr, dtype=float)

    normalizeInRange(arr, 0)
    arr = arr * 255

    # Fill with NaN the zeros to use np.nanmean function
    arr[arr == 0] = np.nan
    mean = np.nanmean(arr)
    arr[arr == np.nan] = 0

    # Calculate thresholds based on mean value and low and high ratios
    min_th = mean * (low_ratio / 255.0) * (high_ratio / low_ratio)
    max_th = mean * (high_ratio / 255.0) * (high_ratio / low_ratio)
    use_processes = True
    num_of_processes = 8
    if use_processes:
        # Using processes
        edges_X = RawArray("d", edges.shape[0] * edges.shape[1])
        edges_X_np = np.frombuffer(edges_X).reshape(edges.shape)
        np.copyto(edges_X_np, edges)

        arr_X = RawArray("d", arr.shape[0] * arr.shape[1])
        arr_X_np = np.frombuffer(arr_X).reshape(arr.shape)
        np.copyto(arr_X_np, arr)

        diff = math.floor(arr.shape[0] / num_of_processes)

        with Pool(
            processes=num_of_processes,
            initializer=mph.hysterisis_worker,
            initargs=(arr_X, edges_X, edges.shape, min_th, max_th, diff),
        ) as pool:
            result = pool.map(mph.hysteresis_threaded, range(0, edges.shape[0], diff))

        edges = edges_X_np
    else:
        # Serial
        for i in range(0, edges.shape[0]):
            for j in range(0, edges.shape[1]):
                if arr[i][j] >= max_th:
                    edges[i][j] = 255
                elif arr[i][j] < min_th:
                    edges[i][j] = 0

        changes = 1
        while changes > 0:
            changes = 0
            for i in range(1, edges.shape[0] - 1):
                for j in range(1, edges.shape[1] - 1):
                    if (
                        (arr[i][j] >= min_th)
                        and (arr[i][j] < max_th)
                        and (edges[i][j] == 0)
                    ):
                        if np.sum(edges[i - 1 : i + 2, j - 1 : j + 2] > 0):
                            edges[i][j] = 255
                            changes += 1

    return edges

# ====================MAIN FUNCTIONS========================

# ==================TEST PLOTS==============================
# def test_noise_reduction(img, blurred):
#     """
#     Plots noise reduction
#     :param img: Original image
#     :param blurred: Blurred image
#     """
#     # Plot images
#     fig = plt.figure("Noise Reduction")
#     plt.subplot(1, 2, 1)
#     plt.axis("off")
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Original Image")
#     plt.subplot(1, 2, 2)
#     plt.axis("off")
#     plt.imshow(cv2.cvtColor(blurred.astype(np.float32)/255.0, cv2.COLOR_BGR2RGB))
#     plt.title("My Blurred Image")

# def test_image_gradient(dx ,dy, magni):
#     """
#     Plots Magnitude
#     :param magni: array with magnitude for every pixel
#     """
#     # Plot images
#     fig2 = plt.figure("Image Gradient")
#     plt.subplot(1, 2, 1)
#     plt.axis("off")
#     plt.imshow(dx, cmap="gray", vmin=-1, vmax=1)
#     plt.title("dI/dx")
#     plt.subplot(1, 2, 2)
#     plt.axis("off")
#     plt.imshow(dy, cmap="gray", vmin=-1, vmax=1)
#     plt.title("dI/dy")

#     fig3 = plt.figure("Image Magnitude")
#     plt.plot()
#     plt.axis("off")
#     plt.imshow(magni, cmap="gray", vmin=0, vmax=1)
#     plt.title("Magnitude")

# def test_edge_thinning(maxima):
#     """
#     Plots image after edge thinning
#     :param maxima: array with local maxima
#     """
#     fig4 = plt.figure("Non-maximum Suppression Test")
#     plt.plot()
#     plt.axis("off")
#     plt.imshow(maxima, cmap="gray")
#     plt.title("Edge Thinning")

# def test_strong_edges(my_edges, opencv_edges):
#     """
#     Plots final edge detection compared with openCV Canny method
#     :param my_edges: array with algorithm edges
#     :param opencv_edges: array with openCV edges
#     """
#     # Plot images
#     fig5 = plt.figure("Edge Detection")
#     plt.subplot(1, 2, 1)
#     plt.axis("off")
#     plt.imshow(my_edges, cmap="gray", vmin=0, vmax=255)
#     plt.title("My Edge Detection")
#     plt.subplot(1, 2, 2)
#     plt.axis("off")
#     plt.imshow(opencv_edges, cmap="gray", vmin=0, vmax=255)
#     plt.title("OpenCV Edges")
# # ==================TEST PLOTS==============================

# if __name__ == "__main__":
#     # use_processes = False
#     img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
#     print("Input Image: " + input_image)
#     print("Kernel Size: " + str(kernel_size))
#     print("Sigma: " + str(sigma))
#     if use_processes:
#         print("[Running Using Multiple Processes, where is possible]")
#     start_time = time.time()

#     print("Step 1: Noise Reduction...")
#     gaussian_kernel = gaussian_kernel_2D(kernel_size, sigma)
#     gau_img = convolution_2D(img, gaussian_kernel, cv2.BORDER_REPLICATE)
#     test_noise_reduction(img, gau_img)

#     print("Step 2: Image Gradient...")
#     gray = cv2.cvtColor(gau_img.astype(np.float32), cv2.COLOR_BGR2GRAY)
#     if use_processes:
#         dx, dy = sobel_threaded_manager(gray)
#     else:
#         dx = sobel_x(gray)
#         dy = sobel_y(gray)
#     magnitude, direction = magnitude_and_direction(dx, dy)
#     test_image_gradient(dx, dy, magnitude)

#     print("Step 3: Non-maximum Suppression...")
#     local_maxima = non_maximum_suppression(magnitude, direction)
#     test_edge_thinning(local_maxima)

#     print("Step 4: Hysteresis Thresholding")
#     print("    ---4.1: Manual Implementation...")
#     my_edges = hysteresis_thresholding(local_maxima, 100, 200)
#     exec_time = time.time() - start_time
#     # print("    ---4.2: OpenCV Implementation...")
#     # edges_auto = cv2.Canny(image=np.uint8(gau_img), threshold1=100, threshold2=200)
#     # test_strong_edges(my_edges, edges_auto)

#     print("\nExecution Time: %s seconds." % "{:.2f}".format(exec_time))
#     plt.show()
