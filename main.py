import cv2
import time
import random
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Pilihan pengguna
print("Pilih mode input:")
print("1. Video")
print("2. Gambar")
mode = input("Masukkan pilihan (1/2): ")

if mode == "1":
    print("Pilih sumber video:")
    print("1. Webcam")
    print("2. File Video (.mp4)")
    video_source = input("Masukkan pilihan (1/2): ")

    if video_source == "1":
        source = 0  # Webcam
    elif video_source == "2":
        source = input("Masukkan path file video (contoh: tes2.mp4): ")
    else:
        print("Pilihan tidak valid.")
        exit()
elif mode == "2":
    image_path = input("Masukkan path gambar (contoh: contoh.jpg): ")
else:
    print("Pilihan tidak valid.")
    exit()

# Konfigurasi
model_path = "yolo12n.pt"
conf_threshold = 0.3
nms_threshold = 0.5


# Inisialisasi model
model = YOLO(model_path)


# Warna per track dan label
track_colors = {}
label_colors = {}

if mode == "2":
    image = cv2.imread(image_path)
    results = model.predict(image, conf=conf_threshold, iou=nms_threshold, verbose=False)[0]

    class_names = model.names
    class_counts = defaultdict(int)
    total_detections = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = class_names[cls_id]

        # Simpan warna unik untuk label jika belum ada
        if label not in label_colors:
            label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))
        color = label_colors[label]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        class_counts[label] += 1
        total_detections += 1

    # Statistik di pojok kiri atas
    y_offset = 20
    cv2.rectangle(image, (10, 5), (180, 90 + len(class_counts) * 20), (0, 0, 0), -1)
    cv2.putText(image, f"Confidence: {conf_threshold:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 20
    cv2.putText(image, f"NMS IoU: {nms_threshold:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += 20
    cv2.putText(image, f"Total: {total_detections}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    y_offset += 20

    for label, count in class_counts.items():
        color = label_colors.get(label, (255, 255, 255))
        cv2.putText(image, f"{label}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20

    cv2.imshow("Hasil Deteksi Gambar", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Inisialisasi video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera/video.")
        exit()

    tracker = DeepSort(max_age=30)
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.predict(frame, conf=conf_threshold, iou=nms_threshold, verbose=False)[0]

        detections = []
        class_names = model.names
        class_counts = defaultdict(int)
        total_detections = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = class_names[cls_id]

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
            class_counts[label] += 1
            total_detections += 1

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cls_id = track.det_class
            label = class_names[cls_id]

            if track_id not in track_colors:
                track_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
            color = track_colors[track_id]

            if label not in label_colors:
                label_colors[label] = color

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0

        # Statistik
        y_offset = 20
        cv2.rectangle(frame, (10, 5), (180, 90 + len(class_counts) * 20), (0, 0, 0), -1)

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"Confidence: {conf_threshold:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"NMS IoU: {nms_threshold:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        cv2.putText(frame, f"Total: {total_detections}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y_offset += 20

        for label, count in class_counts.items():
            color = label_colors.get(label, (255, 255, 255))
            cv2.putText(frame, f"{label}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        # Tampilkan frame
        cv2.imshow("YOLO + DeepSORT", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('+'), ord('w')]:
            conf_threshold = min(1.0, conf_threshold + 0.05)
        elif key in [ord('-'), ord('s')]:
            conf_threshold = max(0.0, conf_threshold - 0.05)
        elif key == ord('e'):
            nms_threshold = min(1.0, nms_threshold + 0.05)
        elif key == ord('d'):
            nms_threshold = max(0.0, nms_threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()
