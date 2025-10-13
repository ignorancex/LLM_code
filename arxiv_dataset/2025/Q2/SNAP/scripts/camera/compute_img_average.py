import cv2
import sys

def compute_img_average(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	return int(img.mean())
if __name__ == '__main__':
	img_path = sys.argv[1]
	print(compute_img_average(img_path))
