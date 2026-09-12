import cv2
import pyttsx3
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Inicializar síntesis de voz
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def hablar(texto):
    """Reproducir texto en voz alta"""
    print(f"🔊 ALERTA: {texto}")
    try:
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Error de sonido: {e}")

# Cargar modelo YOLO
print("📊 Cargando modelo YOLO8...")
model = YOLO('yolov8n.pt')

# Objetos prioritarios para detectar
OBJETOS_PRIORITARIOS = {
    0: 'Persona', 1: 'Bicicleta', 2: 'Coche', 5: 'Autobús', 7: 'Camión',
    11: 'Semáforo', 13: 'Señal', 15: 'Banco', 56: 'Silla', 57: 'Sofá',
    59: 'Cama', 60: 'Mesa'
}

# Umbrales de confianza
CONFIANZA_MINIMA = 0.5
AREA_MINIMA = 0.05

# Control de avisos
ultimo_aviso_tiempo = {}
intervalo_aviso = 2.5
advertencias_peligro = defaultdict(float)

def detectar_color_semaforo(frame, x1, y1, x2, y2):
    """Detectar si el semáforo está en rojo, amarillo o verde"""
    roi = frame[y1:y2, x1:x2]
    
    # Convertir a HSV para mejor detección de colores
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de colores
    # Rojo
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Verde
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # Amarillo
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Crear máscaras
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Contar píxeles de cada color
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
    yellow_count = cv2.countNonZero(mask_yellow)
    
    if red_count > green_count and red_count > yellow_count:
        return "rojo"
    elif green_count > red_count and green_count > yellow_count:
        return "verde"
    elif yellow_count > 0:
        return "amarillo"
    else:
        return "desconocido"

def obtener_posicion(centro_x, ancho_pantalla):
    """Determinar posición izquierda/centro/derecha"""
    if centro_x < ancho_pantalla * 0.35:
        return "a la izquierda"
    elif centro_x > ancho_pantalla * 0.65:
        return "a la derecha"
    else:
        return "al frente"

def obtener_distancia(proporcion_area):
    """Determinar distancia basada en el área"""
    if proporcion_area > 0.30:
        return "muy cerca"
    elif proporcion_area > 0.15:
        return "cerca"
    elif proporcion_area > 0.08:
        return "a distancia de precaución"
    else:
        return "a distancia"

def es_peligro(cls_id, nombre_objeto, proporcion_area):
    """Determinar si es un peligro inmediato"""
    objetos_peligro = [0, 2, 5, 7, 11]  # Persona, Coche, Autobús, Camión, Semáforo
    
    if cls_id in objetos_peligro:
        if cls_id == 11:  # Semáforo
            return True
        elif proporcion_area > 0.15:  # Vehículos o personas cerca
            return True
    
    return False

# Inicializar cámara
print("📷 Iniciando cámara...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    hablar("Error: No se puede acceder a la cámara")
    exit()

hablar("Copiloto iniciado. Sistema listo")
print("✅ Sistema iniciado correctamente\n")

# Loop principal
frame_count = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame")
            break

        frame_count += 1
        
        # Invertir espejo para coordenadas correctas
        frame = cv2.flip(frame, 1)

        ancho_pantalla = frame.shape[1]
        alto_pantalla = frame.shape[0]

        # Realizar detección
        results = model(frame, stream=True, verbose=False, conf=CONFIANZA_MINIMA)

        objetos_detectados = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confianza = float(box.conf[0])
                
                if cls_id in OBJETOS_PRIORITARIOS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    centro_x = (x1 + x2) // 2
                    centro_y = (y1 + y2) // 2
                    
                    area_objeto = (x2 - x1) * (y2 - y1)
                    area_total = ancho_pantalla * alto_pantalla
                    proporcion_area = area_objeto / area_total

                    if proporcion_area > AREA_MINIMA:
                        nombre_objeto = OBJETOS_PRIORITARIOS[cls_id]
                        posicion = obtener_posicion(centro_x, ancho_pantalla)
                        distancia = obtener_distancia(proporcion_area)
                        
                        # Información básica
                        info_objeto = {
                            'nombre': nombre_objeto,
                            'posicion': posicion,
                            'distancia': distancia,
                            'clase': cls_id,
                            'confianza': confianza,
                            'area': proporcion_area,
                            'coords': (x1, y1, x2, y2)
                        }

                        # Detectar color de semáforo
                        if cls_id == 11:  # Semáforo
                            color = detectar_color_semaforo(frame, x1, y1, x2, y2)
                            info_objeto['color'] = color
                            mensaje = f"Semáforo en {color} {posicion}"
                            color_rect = (0, 0, 255) if color == "rojo" else (0, 255, 0) if color == "verde" else (0, 165, 255)
                        else:
                            mensaje = f"{nombre_objeto} {posicion}, {distancia}"
                            color_rect = (0, 255, 0)

                        objetos_detectados.append(info_objeto)

                        # Avisar si es peligro
                        tiempo_actual = time.time()
                        key_aviso = f"{cls_id}_{nombre_objeto}"
                        
                        if es_peligro(cls_id, nombre_objeto, proporcion_area):
                            if tiempo_actual - ultimo_aviso_tiempo.get(key_aviso, 0) > intervalo_aviso:
                                hablar(mensaje)
                                ultimo_aviso_tiempo[key_aviso] = tiempo_actual
                        else:
                            # Avisos normales con intervalo mayor
                            if tiempo_actual - ultimo_aviso_tiempo.get(key_aviso, 0) > intervalo_aviso * 1.5:
                                hablar(mensaje)
                                ultimo_aviso_tiempo[key_aviso] = tiempo_actual

                        # Dibujar en frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
                        cv2.putText(frame, mensaje, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)
                        cv2.circle(frame, (centro_x, centro_y), 5, (0, 0, 255), -1)

        # Mostrar información en pantalla
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Objetos: {len(objetos_detectados)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Presiona 'Q' para salir", (10, alto_pantalla - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar video
        cv2.imshow('🚨 Copiloto - Sistema de Visión Asistida', frame)

        # Salir con 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            hablar("Sistema detenido")
            break

except KeyboardInterrupt:
    print("\n⚠️ Sistema interrumpido por el usuario")
    hablar("Sistema detenido")
except Exception as e:
    print(f"❌ Error: {e}")
    hablar(f"Error del sistema: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Recursos liberados. Adiós!")
