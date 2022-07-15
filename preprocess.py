import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

def crop_contour_retinal_image(image, image_size=(256, 256), plot=False):
    image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayed = cv2.GaussianBlur(grayed, (5, 5), 0)
    thresh_image = cv2.erode(grayed, None, iterations=2)
    thresh_image = cv2.dilate(thresh_image, None, iterations=2)
    
    # find contours
    contours = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)

    # extract bounding coords of largest contour
    extreme_pnts_left = tuple(c[c[:, :, 0].argmin()][0])
    extreme_pnts_right = tuple(c[c[:, :, 0].argmax()][0])
    extreme_pnts_top = tuple(c[c[:, :, 1].argmin()][0])
    extreme_pnts_bot = tuple(c[c[:, :, 1].argmax()][0])

    # crop image
    new_image = grayed[extreme_pnts_top[1]:extreme_pnts_bot[1], extreme_pnts_left[0]:extreme_pnts_right[0]]

    # apply adaptive equaliation of image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(new_image)

    # apply image normalization
    mask = np.zeros((256,256))
    normalized = cv2.normalize(equalized, mask, 0, 255, cv2.NORM_MIN_MAX)

    new_image = cv2.resize(normalized, image_size, interpolation=cv2.INTER_AREA)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original - {image.shape}")
        plt.axis('off')
        plt.grid(False)
        
        # ---------------------------------- #

        plt.subplot(1, 2, 2)
        plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Preprocessed - {new_image.shape}")
        plt.axis('off')
        plt.grid(False)

        plt.show()

    return new_image
