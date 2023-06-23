import cv2
import numpy as np

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

classes = {0: "background", 1: "aeroplane", 2: "bicycle",
           3: "bird", 4: "boat", 5: "Botella", 6: "bus",
           7: "car", 8: "cat", 9: "chair", 10: "cow",
           11: "diningtable", 12: "dog", 13: "horse",
           14: "motorbike", 15: "Persona", 16: "pottedplant",
           17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

net = cv2.dnn.readNetFromCaffe(prototxt, model)

cap = cv2.VideoCapture('video2.mp4')

# Parámetros para la detección de grupos
min_group_size = 3  # Mínimo número de personas para considerarlo un grupo
max_distance = 120   # Máxima distancia permitida entre personas para formar un grupo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))

    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

    net.setInput(blob)
    detections = net.forward()

    # Lista para almacenar las coordenadas de las personas detectadas
    people_coords = []

    for detection in detections[0, 0]:
        class_id = int(detection[1])
        if class_id == 15:  # Solo se consideran las detecciones de personas
            confidence = detection[2]
            if confidence > 0.5:
                box = detection[3:7] * np.array([width, height, width, height])
                x_start, y_start, x_end, y_end = box.astype(int)

                # Guardar las coordenadas de la persona
                people_coords.append((x_start, y_start, x_end, y_end))

                # Dibujar el cuadro delimitador y etiquetas
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, "Conf: {:.2f}".format(confidence * 100), (x_start, y_start - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, classes[class_id], (x_start, y_start - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Verificar si hay grupos de personas cercanas
    if len(people_coords) >= min_group_size:
        for i in range(len(people_coords) - 1):
            for j in range(i + 1, len(people_coords)):
                x1, y1, _, _ = people_coords[i]
                x2, y2, _, _ = people_coords[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if distance < max_distance:
                    # Dibujar línea entre las personas del grupo
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Mostrar alerta de grupo detectado
                    cv2.putText(frame, "Grupo Detectado", (x1, y1 - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()