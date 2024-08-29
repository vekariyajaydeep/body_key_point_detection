from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_pose(video_path):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    print(f"VIDEO: {video_path}")
   
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (350, 600))

        results = pose.process(img)
        
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )
        cv2.imshow("Pose Estimation", img)

        h, w, c = img.shape   
        opImg = np.zeros([h, w, c])  
        opImg.fill(255)  

        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )
        cv2.imshow("Extracted Pose", opImg)

        print(results.pose_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' in request.files:
        video = request.files['video']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        print(f"Video: {video}, Videopath: {video_path}")
        detect_pose(video_path)
        return render_template('again.html')

if __name__=="__main__":
    app.run(debug=True, port=94)