from ultralytics import YOLO
import cv2

#สมการ Linear Regression 
EQUATIONS = {
    "FrontHoop": (-0.0034, 828.7),
    "LeftHoop": (-0.0029, 799.54),
    "RightHoop": (-0.0024, 760.59),
}

#โหลดโมเดล YOLO,
model_path = "D:/TrackMe02/train/weights/best.pt"
model = YOLO(model_path)

#เปิดกล้อง,
cap = cv2.VideoCapture(0)  # ใช้กล้องหลัก
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#ตรวจสอบว่ากล้องเปิดได้หรือไม่,
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

#อ่านข้อมูลกล้อง,
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 30

#วนลูปรับภาพจากกล้อง
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    results = model(gray_frame)

    detected = False
    type_hoop = "None"
    bbox_size = "0 x 0"
    confidence = "0.00"
    distance = "0.00"

    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            if conf < 0.80:
                continue  # ข้ามกล่องที่ confidence < 85%

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            type_hoop = model.names[cls]
            width, height = x2 - x1, y2 - y1
            bbox_size = f"{width} x {height}"
            confidence = f"{conf:.2f}"
            detected = True

            pixel_area = width * height
            if type_hoop in EQUATIONS:
                m, b = EQUATIONS[type_hoop]
                distance_val = m * pixel_area + b
            else:
                distance_val = 0
            distance = f"{distance_val:.2f} cm"

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 10, 255), 3)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (90, 10, 255), 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (90, 10, 255), 2)

    info_box_x, info_box_y = frame_width - 600, 70
    info_box_w, info_box_h = 600, 200
    cv2.rectangle(frame, (info_box_x, info_box_y), (info_box_x + info_box_w, info_box_y + info_box_h), (0, 0, 0), -1)

    text_color = (255, 255, 255)
    font_scale = 1.0
    font_thickness = 3

    cv2.putText(frame, f"Type   : {type_hoop}", (info_box_x + 10, info_box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    cv2.putText(frame, f"Pixel Bounding Box: {bbox_size}", (info_box_x + 10, info_box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    cv2.putText(frame, f"Confidence: {confidence}", (info_box_x + 10, info_box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    cv2.putText(frame, f"Distance: {distance}", (info_box_x + 10, info_box_y + 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    cv2.putText(frame, f"Detection: [ {detected} ]", (info_box_x + 10, info_box_y + 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    frame_resized = cv2.resize(frame, (int(frame_width * 0.7), int(frame_height * 0.7)))
    cv2.imshow("YOLO Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#ปิดกล้อง
cap.release()
cv2.destroyAllWindows()