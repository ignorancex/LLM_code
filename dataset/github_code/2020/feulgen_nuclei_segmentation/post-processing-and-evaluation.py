import cv2
import numpy as np


if __name__ == '__main__':

    img = cv2.imread('images/input_image.png')
    out = cv2.imread('images/network_output.png')
    gt = cv2.imread('images/ground_truth.png')

    pixels_gt = gt[:, :, 0]
    image_visualization = np.zeros(
        (img.shape[0], img.shape[1], img.shape[2]), np.uint8)

    out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out_gray = cv2.GaussianBlur(out_gray, (5, 5), cv2.BORDER_DEFAULT)

    ret, th1 = cv2.threshold(out_gray, 230, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(th1, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

    # Generating image for visualization purpose
    image_visualization[pixels_gt == 255] = (255, 0, 0)  # Blue = Ground Truth
    image_visualization[opening == 255] = (0, 255, 0)  # Red = Predictions

    edges = cv2.Canny(opening, 100, 200)

    # # You can apply a dilation operation on edges to make it more thick
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # erosion = cv2.dilate(edges, kernel, iterations=1)
    # img[edges == 255] = (0, 0, 255)

    img[edges == 255] = (0, 0, 255)

    # Calculating IoU
    intersection = cv2.bitwise_and(opening, pixels_gt)
    union = cv2.bitwise_or(opening, pixels_gt)

    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU: " + str(iou_score))

    cv2.namedWindow("IoU", cv2.WINDOW_NORMAL)
    cv2.imshow("IoU", image_visualization)

    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
