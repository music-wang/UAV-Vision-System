import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\deepseek harness work space\UAV-Vision-System\runs\2025h_synth\weights\best.onnx")

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # 推理（置信度设置为 0.5）
    results = model(frame, conf=0.5, verbose=False)
    
    # 画框
    annotated_frame = results[0].plot()
    
    # 显示
    cv2.imshow("UAV YOLO Detection", annotated_frame)
    
    # 重点：检测窗口是否被鼠标点了 X 关掉
    if cv2.getWindowProperty("UAV YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

    # 依然保留按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 彻底释放资源
cap.release()
cv2.destroyAllWindows()