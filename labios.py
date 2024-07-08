import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    return annotated_image

def refine_lip_mask(lip_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lip_mask = cv2.erode(lip_mask, kernel, iterations=1)
    lip_mask = cv2.dilate(lip_mask, kernel, iterations=2)
    lip_mask = cv2.GaussianBlur(lip_mask, (3, 3), 0)
    return lip_mask

def apply_lip_color(image, lip_mask, color):
    # Convertir la imagen original y la máscara a espacio de color LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB)[0][0]

    # Aplicar la máscara para cambiar solo los píxeles de los labios
    l, a, b = cv2.split(lab_image)
    l = np.where(lip_mask == 255, lab_color[0], l)
    a = np.where(lip_mask == 255, lab_color[1], a)
    b = np.where(lip_mask == 255, lab_color[2], b)

    # Combinar los canales y convertir de nuevo a BGR
    lab_colored_lips = cv2.merge([l, a, b])
    colored_lips_image = cv2.cvtColor(lab_colored_lips, cv2.COLOR_LAB2BGR)
    
    return colored_lips_image

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create a FaceLandmarker object.
model_path = '/Users/developerkrika/Desktop/make up/Virtual_Makeup_opencv/face_landmarker_v2_with_blendshapes.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image_path = 'sample/face.png'  # Cambia esto a la ruta de tu imagen local
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(mp_image)

# STEP 5: Define the indices for different parts of the lips.
indices = {
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
}

regions = {region: [] for region in indices}

if detection_result.face_landmarks:
    for face_landmarks in detection_result.face_landmarks:
        for region, region_indices in indices.items():
            regions[region] = [(int(face_landmarks[i].x * image.shape[1]), int(face_landmarks[i].y * image.shape[0])) for i in region_indices]
            print(f"{region.capitalize()} Points: ", regions[region])

# Crear una máscara para la región de los labios.
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Dibujar los puntos de los labios en la máscara.
for region, points in regions.items():
    for point in points:
        cv2.circle(mask, point, 1, 255, -1)  # Blanco para los puntos de los labios

# Crear contornos para los labios.
lip_contour_outer = np.array(regions["lipsUpperOuter"] + regions["lipsLowerOuter"][::-1], dtype=np.int32)
lip_contour_inner = np.array(regions["lipsUpperInner"] + regions["lipsLowerInner"][::-1], dtype=np.int32)

# Rellenar las regiones de los labios.
cv2.fillPoly(mask, [lip_contour_outer], 255)
cv2.fillPoly(mask, [lip_contour_inner], 0)

# Refinar la máscara de los labios
refined_mask = refine_lip_mask(mask)

# Color deseado para los labios (BGR).
lip_color = [77, 38, 147]  # Ajustar según sea necesario

# Aplicar color a los labios segmentados.
colored_lips_image = apply_lip_color(image, refined_mask, lip_color)

# Mostrar la imagen original y la imagen con labios coloreados.
cv2.imshow('Original Image', image)
cv2.imshow('Colored Lips', colored_lips_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
