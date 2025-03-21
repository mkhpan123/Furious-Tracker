from flask import Flask, Response, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/process_frame', methods=['POST'])
def video_feed():
    try:
        file = request.files['frame']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        return Response(str(e), status=400)

if __name__ == '__main__':
    app.run(debug=True)