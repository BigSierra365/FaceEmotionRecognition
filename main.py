"""
SISTEMA DE RASTREO BIOMÉTRICO Y DETECCIÓN DE EMOCIONES (HUD PRO)

Este script implementa un sistema de visión artificial en tiempo real que combina
la precisión de MediaPipe para el rastreo facial y la potencia de YOLO11 para
la clasificación de microexpresiones emocionales.

Librerías principales:
- OpenCV (cv2): Interfaz de cámara y renderizado de gráficos.
- MediaPipe (mp): Generación de la malla facial (Face Mesh) en 3D.
- YOLO (ultralytics): Inferencia de redes neuronales para detección de emociones.
- NumPy: Procesamiento eficiente de matrices y coordenadas.

Flujo de ejecución:
1. Carga de infraestructura: Inicialización del modelo YOLO (.pt) y servicios de MediaPipe.
2. Captura de vídeo: Apertura del flujo de cámara y preprocesamiento de cuadros (flip/color).
3. Análisis de malla: Detección de puntos de referencia faciales y cálculo de caja delimitadora.
4. Extracción y Clasificación: Recorte del rostro en alta fidelidad y predicción de emoción mediante YOLO.
5. Renderizado HUD: Dibujo de la careta "holográfica" y visualización de telemetría emocional.
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Definición del esquema de colores y traducción de etiquetas para la interfaz gráfica
EMOTION_THEMES = {
    'angry': {'color': (0, 0, 255), 'name': 'Ira'},
    'fearful': {'color': (255, 0, 255), 'name': 'Miedo'},
    'happy': {'color': (0, 255, 0), 'name': 'Felicidad'},
    'neutral': {'color': (200, 200, 200), 'name': 'Neutral'},
    'sad': {'color': (255, 0, 0), 'name': 'Tristeza'},
    'surprised': {'color': (0, 255, 255), 'name': 'Sorpresa'}
}
DEFAULT_COLOR = (255, 255, 255)

# Bloque de inicialización del modelo de inteligencia artificial pre-entrenado
try:
    yolo_model = YOLO("feryolo-11x-64.pt")
    print("Modelo YOLO11 de Emociones cargado con éxito.")
except Exception as e:
    print(f"Error crítico: No se encontró o no se pudo cargar 'feryolo-11x-64.pt'.")
    print(f"Detalles del error: {e}")
    exit()

# Configuración del motor de rastreo facial de alta densidad de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Inicialización de la captura del hardware de vídeo
cap = cv2.VideoCapture(0)

print("Sistema de Rastreo HUD iniciado. Presiona 'q' para salir.")

# Bucle principal de procesamiento de vídeo en tiempo real
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Efecto espejo para interacción natural
    h, w, _ = frame.shape
    
    # Capa secundaria para el dibujo de la interfaz (HUD)
    hud_overlay = np.zeros_like(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamiento de la imagen para extraer los puntos de referencia de la cara
    mesh_results = face_mesh.process(rgb_frame)
    
    dominant_emotion = 'neutral'
    emotion_conf = 0.0
    hud_color = DEFAULT_COLOR
    text_position = None

    # Lógica aplicada si se detecta al menos un rostro en el cuadro
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            
            # Cálculo de los límites geométricos del rostro para el recorte posterior
            x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
            
            x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
            
            # Aplicación de margen de seguridad para asegurar la visibilidad total del rostro
            margin = 30
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            # Extracción del segmento de imagen que contiene el rostro para análisis de IA
            face_crop = frame[y_min:y_max, x_min:x_max]

            # Inferencia mediante el modelo neuronal para determinar la emoción predominante
            if face_crop.size > 0:
                yolo_results = yolo_model.predict(face_crop, verbose=False)
                
                if yolo_results[0].probs is not None:
                    probs = yolo_results[0].probs
                    cls_id = probs.top1
                    emotion_conf = float(probs.top1conf)
                    
                    # Filtro de confianza para evitar detecciones erráticas o ruido
                    if emotion_conf > 0.45:
                        dominant_emotion = yolo_model.names[cls_id]
                        theme = EMOTION_THEMES.get(dominant_emotion.lower())
                        if theme:
                            hud_color = theme['color']
                            emotion_display_name = theme['name']
                        else:
                            emotion_display_name = dominant_emotion.capitalize()

            # Renderizado de las líneas de la careta sobre la capa del HUD
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                p1_idx = connection[0]
                p2_idx = connection[1]
                
                pt1 = (int(face_landmarks.landmark[p1_idx].x * w), int(face_landmarks.landmark[p1_idx].y * h))
                pt2 = (int(face_landmarks.landmark[p2_idx].x * w), int(face_landmarks.landmark[p2_idx].y * h))
                
                cv2.line(hud_overlay, pt1, pt2, hud_color, 1, cv2.LINE_AA)
            
            # Definición del punto espacial donde se mostrará la etiqueta de texto
            text_position = (int(face_landmarks.landmark[151].x * w), int(face_landmarks.landmark[151].y * h))

    # Unión de la imagen original con la capa HUD mediante peso de transparencia
    final_output = cv2.addWeighted(frame, 0.7, hud_overlay, 1.0, 0)
    
    # Dibujo de los indicadores de texto y telemetría si hay una detección válida
    if text_position and hud_color != DEFAULT_COLOR:
        prediction_text = f"DETECCION: {emotion_display_name} [{int(emotion_conf*100)}%]"
        text_x = text_position[0] - 120
        text_y = text_position[1] - 30
        
        cv2.putText(final_output, prediction_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, hud_color, 2, cv2.LINE_AA)
        cv2.line(final_output, (text_x, text_y + 10), (text_x + 250, text_y + 10), hud_color, 1)

    # Mensaje de estado cuando no hay rostros detectados en el área de escaneo
    elif not text_position:
         cv2.putText(final_output, "Escaneando...", (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEFAULT_COLOR, 2)

    # Lanzamiento de la ventana de visualización al usuario
    cv2.imshow("Face Tracking & Emotion Detection HUD - PRO", final_output)

    # Gestión de salida del programa mediante entrada de teclado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberación de recursos de hardware y cierre de ventanas
cap.release()
cv2.destroyAllWindows()