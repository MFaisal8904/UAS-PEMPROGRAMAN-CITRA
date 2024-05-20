import numpy as np
import cv2

def translate(image, x, y):
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	return shifted

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def rotate(image, angle, center = None, scale = 1.0):
	(h, w) = image.shape[:2]

	if center is None:
		center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	
	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		
		r = width / float(w)
		dim = (width, int(h * r))

	
	resized = cv2.resize(image, dim, interpolation = inter)

	return resized
