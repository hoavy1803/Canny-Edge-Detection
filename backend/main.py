from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import math
import numpy as np
import io
import canny

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_edges(image, kernelsize, low, high):
    # Đọc ảnh từ dữ liệu nhị phân
    nparr = np.frombuffer(image, np.uint8)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    kernel_size = kernelsize
    sigma = math.floor(kernel_size / 2)

    use_processes = True
    # img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    gaussian_kernel = canny.gaussian_kernel_2D(kernel_size, sigma)
    gau_img = canny.convolution_2D(input_img, gaussian_kernel, cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(gau_img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if use_processes:
        dx, dy = canny.sobel_threaded_manager(gray)
    else:
        dx = canny.sobel_x(gray)
        dy = canny.sobel_y(gray)
    magnitude, direction = canny.magnitude_and_direction(dx, dy)

    local_maxima = canny.non_maximum_suppression(magnitude, direction)

    my_edges = canny.hysteresis_thresholding(local_maxima, low, high)

    # Chuyển đổi ảnh tìm cạnh thành định dạng PNG
    _, buffer = cv2.imencode(".png", my_edges)
    return buffer.tobytes()


@app.get("/")
async def hello():
    return {"Hello World"}


# Xử lý kết quả và trả về phản hồi

@app.post("/edge-detection")
async def edge_detection(
    file: UploadFile = File(...),
    selectedKernel: int = Form(...),
    selectedLowThreshold: int = Form(...),
    selectedHighThreshold: int = Form(...),
):
    # Đọc dữ liệu ảnh từ tệp tin
    image_data = await file.read()

    # Xử lý tìm cạnh
    edge_image = detect_edges(
        image_data, selectedKernel, selectedLowThreshold, selectedHighThreshold
    )

    # Trả về kết quả dưới dạng tệp tin ảnh
    return StreamingResponse(io.BytesIO(edge_image), media_type="image/png")
