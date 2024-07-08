import cv2
import mediapipe as mp
import numpy as np

# Landmarks of features from mediapipe
face_points = {
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
}

# Initialize mediapipe functions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def read_landmarks(image: np.array):
    landmark_cordinates = {}
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            for idx, landmark in enumerate(face_landmarks):
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0]
                )
                if landmark_px:
                    landmark_cordinates[idx] = landmark_px
    return landmark_cordinates

def get_lip_mask(image, landmarks, lip_points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([landmarks[idx] for idx in lip_points])
    cv2.fillPoly(mask, [points], 255)

    # Refine the mask using advanced morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.erode(mask, kernel, iterations=1)  # Reduce iterations for a thinner mask
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

def makeup_video(lip_color):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = read_landmarks(frame)
        if landmarks:
            lip_mask = get_lip_mask(frame, landmarks, face_points["LIP_UPPER"] + face_points["LIP_LOWER"])
            frame_with_makeup = apply_lip_color(frame, lip_mask, lip_color)
            cv2.imshow('Makeup', frame_with_makeup)
        else:
            cv2.imshow('Makeup', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    lip_color = [0, 0, 100]  # Dark red in BGR
    makeup_video(lip_color)
