import cv2
import argparse
import numpy as np
from utils import *

# features to add makeup
face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
]

# change the color of features
colors_map = {
    # upper lip and lower lips
    "LIP_UPPER": [50, 0, 100],  # Dark Red in BGR
    "LIP_LOWER": [50, 0, 100],  # Dark Red in BGR
}

def fill_contour(image, face_landmarks, points, color):
    contour_points = []
    for idx in points:
        if idx in face_landmarks:
            contour_points.append(face_landmarks[idx])
    if contour_points:
        contour_points = np.array(contour_points, dtype=np.int32)
        cv2.fillPoly(image, [contour_points], color)

def add_smoothing(mask, kernel_size=(7, 7), sigma=5):
    return cv2.GaussianBlur(mask, kernel_size, sigma)

def main(image_path):
    # extract required facial points from face_elements
    face_connections = [face_points[idx] for idx in face_elements]
    # extract corresponding colors for each facial features
    colors = [colors_map[idx] for idx in face_elements]
    # read image
    image = cv2.imread(image_path)
    # create an empty mask-like image
    mask = np.zeros_like(image)
    # extract facial landmarks
    face_landmarks = read_landmarks(image=image)
    
    # Create a separate lip mask and fill with color
    lip_mask = np.zeros_like(image)
    fill_contour(lip_mask, face_landmarks, face_points["LIP_UPPER"], color=colors_map["LIP_UPPER"])  # Fill upper lip with color
    fill_contour(lip_mask, face_landmarks, face_points["LIP_LOWER"], color=colors_map["LIP_LOWER"])  # Fill lower lip with color

    # Apply smoothing to the lip mask
    smoothed_lip_mask = add_smoothing(lip_mask, kernel_size=(7, 7), sigma=5)

    # Combine the original image with the smoothed lip mask using a mask
    lip_mask_gray = cv2.cvtColor(smoothed_lip_mask, cv2.COLOR_BGR2GRAY)
    _, lip_mask_binary = cv2.threshold(lip_mask_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Apply lip color using the binary mask
    color_image = np.zeros_like(image)
    color_image[:] = colors_map["LIP_UPPER"]  # Assuming both upper and lower lip colors are the same
    lip_area_colored = cv2.bitwise_and(color_image, color_image, mask=lip_mask_binary)
    inverse_lip_mask = cv2.bitwise_not(lip_mask_binary)
    original_lip_area = cv2.bitwise_and(image, image, mask=inverse_lip_mask)
    output = cv2.add(original_lip_area, lip_area_colored)

    # Adjust the intensity of the lip color
    output = cv2.addWeighted(image, 0.7, output, 0.3, 0)

    # Concatenate images side by side
    combined_image = cv2.hconcat([image, output])
    
    # display the images in a single window
    show_image(combined_image, msg='Original and Mask with Lip Fill')

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Image to add Facial makeup ")
    # add image path as argument
    parser.add_argument("--img", type=str, help="Path to the image.")
    args = parser.parse_args()
    main(args.img)
