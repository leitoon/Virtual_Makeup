import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def show_image(image: np.array, msg: str = "Loaded Image"):
    cv2.imshow(msg, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_landmarks(image: np.array):
    landmark_coordinates = {}
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            for idx, landmark in enumerate(face_landmarks):
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0]
                )
                if landmark_px:
                    landmark_coordinates[idx] = landmark_px
    return landmark_coordinates

def get_lip_points(landmarks):
    indices = {
        "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    }

    upper_lip_points = [landmarks[idx] for idx in indices["lipsUpperOuter"] + indices["lipsUpperInner"]]
    lower_lip_points = [landmarks[idx] for idx in indices["lipsLowerOuter"] + indices["lipsLowerInner"]]

    return upper_lip_points, lower_lip_points

def get_lip_mask(image, upper_lip_points, lower_lip_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    upper_lip_points = np.array(upper_lip_points)
    lower_lip_points = np.array(lower_lip_points)
    cv2.fillPoly(mask, [upper_lip_points], 255)
    cv2.fillPoly(mask, [lower_lip_points], 255)

    # Refine the mask using advanced morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask = cv2.erode(mask, kernel, iterations=3)  # Increase iterations for a thinner mask
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 1)  # Reduce kernel size and sigma for less blur

    return mask

def apply_lip_color(image, lip_mask, desired_color):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    desired_color_hsv = cv2.cvtColor(np.uint8([[desired_color]]), cv2.COLOR_BGR2HSV)[0][0]

    lip_mask_binary = (lip_mask > 0).astype(np.uint8) * 255
    avg_color = cv2.mean(image, mask=lip_mask_binary)[:3]
    avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]

    diff_h = np.subtract(desired_color_hsv[0], avg_color_hsv[0], dtype=np.int16)
    diff_h = (diff_h + 180) % 180 - 180  # Ensure hue difference is in range [-180, 180]

    diff_s = desired_color_hsv[1] / (avg_color_hsv[1] + 1e-5)
    diff_v = desired_color_hsv[2] / (avg_color_hsv[2] + 1e-5)

    h = (h.astype(np.int16) + diff_h) % 180
    h[h < 0] += 180  # Ensure h is in range [0, 179]
    s = np.clip(s * diff_s, 0, 255).astype(np.uint8)
    v = np.clip(v * diff_v, 0, 255).astype(np.uint8)

    img_hsv = cv2.merge([h.astype(np.uint8), s, v])
    new_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Create a mask for blending
    mask_blend = cv2.merge([lip_mask_binary] * 3) / 255.0

    new_image = (image * (1 - mask_blend) + new_image * mask_blend).astype(np.uint8)
    return new_image

def makeup_image(input_image_path, output_image_path, lip_color):
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not read image from {input_image_path}")
        return

    landmarks = read_landmarks(image)
    if not landmarks:
        print("Error: No landmarks detected.")
        return

    upper_lip_points, lower_lip_points = get_lip_points(landmarks)
    lip_mask = get_lip_mask(image, upper_lip_points, lower_lip_points)

    image_with_makeup = apply_lip_color(image, lip_mask, lip_color)

    cv2.imwrite(output_image_path, image_with_makeup)
    print(f"Image saved to {output_image_path}")

    combined_image = cv2.hconcat([image, image_with_makeup])
    show_image(combined_image, 'Original and Makeup')

# Example usage
if __name__ == "__main__":
    lip_color = [0, 0, 139]  # Dark red in BGR
    input_image_path = 'sample/img6.jpeg'
    output_image_path = 'sample/output_img6_with_makeup.jpg'
    makeup_image(input_image_path, output_image_path, lip_color)
