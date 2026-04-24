import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل نموذج أقوى للكشف (النسخة المتوسطة 'm' أدق للأهداف البعيدة من النسخة 'n')
model = YOLO('yolov8m.pt') 

# قائمة الفئات التي نريد تتبعها (تعتمد على الفئات الافتراضية في YOLO أو نموذج مدرب)
TARGET_CLASSES = ['airplane', 'bird', 'drone', 'helicopter']

def draw_radar_ui(frame, detections):
    h, w, _ = frame.shape
    color = (0, 255, 0) # أخضر راداري
    
    # رسم قائمة الأهداف النشطة (Sidebar) على اليمين
    sidebar_w = 300
    cv2.rectangle(frame, (w - sidebar_w, 0), (w, h), (0, 20, 0), -1)
    cv2.line(frame, (w - sidebar_w, 0), (w - sidebar_w, h), color, 2)
    cv2.putText(frame, "ACTIVE TRACKS", (w - sidebar_w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for i, det in enumerate(detections):
        if i > 15: break # حد أقصى للقائمة
        
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls_id = int(det.cls[0])
        label = model.names[cls_id]

        # فلترة النتائج لتشمل الطائرات فقط
        if label.lower() in TARGET_CLASSES or label.lower() == 'bird': # الطيور أحياناً تشبه الدرونات البعيدة
            target_name = "DRONE FPV" if label == 'bird' else label.upper()
            
            # رسم مربع الاستهداف على الشاشة
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{target_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # إضافة معلومات الهدف إلى القائمة الجانبية
            y_pos = 70 + (i * 40)
            cv2.putText(frame, f"ID:{i} | {target_name[:10]} | {conf:.2%}", 
                        (w - sidebar_w + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(frame, (w - sidebar_w + 5, y_pos + 10), (w - 5, y_pos + 10), (0, 50, 0), 1)

    # إضافة لمسة الرادار (الدوائر المركزية)
    center = ( (w - sidebar_w) // 2, h // 2)
    for r in [150, 300, 450]:
        cv2.circle(frame, center, r, (0, 100, 0), 1)
    
    return frame

# إعداد الكاميرا والدقة
cap = cv2.VideoCapture(0)
# محاولة ضبط الكاميرا على أعلى دقة ممكنة
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# جعل النافذة قابلة للتكبير وملء الشاشة
cv2.namedWindow('Military Radar System', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Military Radar System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # تغيير حجم الإطار ليناسب شاشة اللابتوب (مثلاً 1920x1080)
    frame = cv2.resize(frame, (1920, 1080))

    # الكشف باستخدام YOLO
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # رسم الواجهة
    frame = draw_radar_ui(frame, results[0].boxes)

    cv2.imshow('Military Radar System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
