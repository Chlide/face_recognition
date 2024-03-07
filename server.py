from flask import Flask, Response
import cv2

app = Flask(__name__)

# 摄像头索引号（根据实际情况修改）
camera_index = 0

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(camera_index)

def generate_frames():
    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        if not ret:
            break

        # 将帧转换为字节流
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "摄像头视频流服务"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
