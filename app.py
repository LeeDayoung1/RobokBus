from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import os
from openai import OpenAI
from flask_cors import CORS
import base64
import cv2
from deepface import DeepFace
import numpy as np


load_dotenv()  
app = Flask(__name__)
CORS(app)  

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            try:
                # 나이, 성별, 인종을 모두 분석하도록 설정
                result = DeepFace.analyze(frame, actions=['age', 'gender', 'race'], enforce_detection=False)
                
                # 분석 결과 터미널에 출력
                print("DeepFace Analysis Result:", result)
                
                # 분석 결과에서 나이, 성별, 인종을 프레임에 표시
                age = result['age']
                gender = result['gender']
                race = result['dominant_race']
                cv2.putText(frame, f'Age: {age}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Gender: {gender}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Race: {race}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error analyzing frame:", e)
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # img를 축소하여 분석
        frame_resized = cv2.resize(img, (640, 480))
        
        # DeepFace 분석 수행
        result = DeepFace.analyze(frame_resized, actions=['age', 'gender', 'race'], enforce_detection=False)

        # DeepFace 결과가 리스트 형태로 반환
        if isinstance(result, list):
            result = result[0] 
        
        # 분석 결과에서 나이, 성별, 인종을 추출
        age = result['age']
        gender = result['dominant_gender']
        race = result['dominant_race']
        
        #터미널에 전체 분석 결과 출력
        print("DeepFace Analysis Result:", result) 
        
        # JSON 응답 반환
        return jsonify({'age': age, 'gender': gender, 'race': race})
    except Exception as e:
        app.logger.error(f"Error in /analyze_frame: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
