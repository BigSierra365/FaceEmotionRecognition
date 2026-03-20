# 🎭 Face Emotion Recognition + HUD

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/AI-YOLOv11-orange?logo=ultralytics)
![MediaPipe](https://img.shields.io/badge/Tracking-MediaPipe-green?logo=google)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

## 🚀 Descripción / Propósito

**Face Emotion Recognition + HUD** es un sistema avanzado de visión artificial diseñado para el análisis biométrico y la clasificación de microexpresiones faciales en tiempo real. 

El proyecto resuelve el desafío de realizar un seguimiento facial de alta densidad (Face Mesh) mientras se ejecutan modelos de Redes Neuronales Convolucionales (CNN) de última generación para la detección emocional. Utiliza una arquitectura de **Heads-Up Display (HUD)** para superponer telemetría en tiempo real sobre el rostro del usuario, proporcionando una experiencia interactiva y tecnológica de nivel profesional.

Ideal para:
- Análisis de UX/UI mediante reconocimiento de sentimientos.
- Sistemas de monitoreo de atención y Fatiga.
- Integración en cabinas de simulación o asistencia virtual avanzada.

---

## ⚙️ Stack Tecnológico

| Componente | Tecnologías High-End |
| :--- | :--- |
| **IA / Deep Learning** | [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (FER Inference) |
| **Computer Vision** | [OpenCV](https://opencv.org/) (Real-time Rendering & I/O) |
| **Biometría 3D** | [MediaPipe Face Mesh](https://google.github.io/mediapipe/) (468+ Landmarks) |
| **Logic & Data** | Python 3.9+, NumPy (Matrix Optimization) |

---

## ✨ Características Principales

- 🎯 **Rastreo Facial 3D:** Seguimiento preciso de puntos de referencia críticos incluso con movimientos bruscos o cambios de iluminación.
- 🧠 **Inferencia YOLOv11:** Clasificación instantánea de 6 emociones clave: *Felicidad, Tristeza, Ira, Sorpresa, Miedo y Neutral*.
- 🖥️ **HUD Holográfico:** Capa de visualización semi-transparente que sigue la geometría facial del usuario.
- 🚦 **Filtro de Confianza Adaptativo:** Algoritmo dinámico para reducir falsos positivos en la detección emocional (confianza > 45%).
- ⚡ **Optimización CPU/GPU:** Soporte nativo para aceleración por hardware (CUDA) para framerates fluidos.

---

## 🧠 Arquitectura y Lógica

El sistema opera bajo un flujo de procesamiento lineal optimizado para la mínima latencia:

1. **Captura y Pre-procesado:** Ingesta de frames desde hardware (Webcam) y normalización de color (RGB).
2. **Malla Facial (MediaPipe):** Extracción de la topología facial para definir el área de interés (ROI).
3. **Extracción Dinámica (ROI Crop):** Recorte automático del rostro basado en los landmarks detectados para alimentar al modelo de IA.
4. **Clasificación Neural (YOLOv11):** Inferencia sobre el ROI para determinar el estado emocional con una probabilidad asociada.
5. **Composición HUD:** Fusión de la imagen original con la careta geométrica y telemetría mediante técnicas de `weighted blending`.

---

## 🎥 Demostración



https://github.com/user-attachments/assets/8900304e-8276-4564-9fb5-27e0432b01d3



---

## ⚠️ Prerrequisitos de Hardware y Modelos

### Hardware
- **Webcam:** Resolución mínima recomendada de 720p.
- **Procesador:** Intel i5/AMD Ryzen 5 o superior.
- **GPU (Opcional):** NVIDIA con soporte CUDA para obtener >30 FPS.

### Pesos del Modelo
El sistema requiere el archivo de pesos:
- `feryolo-11x-64.pt` (Incluido en el repositorio o descargado automáticamente por Ultralytics).

---

## 💻 Instalación (Plug & Play)

Sigue estos pasos para desplegar el entorno en tu máquina local:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/detector-emociones.git
   cd detector-emociones
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv env
   # Windows:
   .\env\Scripts\activate
   # Linux/macOS:
   source env/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Uso del Sistema

Para iniciar el sistema HUD, asegúrate de tener conectada una cámara y ejecuta:

```bash
python main.py
```

### Controles de Interfaz:
- **Salida:** Presiona la tecla `Q` en cualquier momento para cerrar el sistema de forma segura.
- **Entrada:** Sitúate frente a la cámara; el sistema detectará el rostro y activará automáticamente el HUD de emociones.

---

> [!TIP]
> **Modo Desarrollador:** Puedes ajustar el umbral de confianza modificando la variable `emotion_conf > 0.45` en el archivo `main.py` para adaptar la sensibilidad del modelo a tu entorno.
