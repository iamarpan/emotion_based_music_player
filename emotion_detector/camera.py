import gradio as gd
import onnx
import onnxruntime 
import cv2
import torch
import torch.nn.functional as F
import json
model_path = 'model.onnx'
import numpy as np
import time
import codecs
html = {
        "empty":'''<html>
                <body>Could not detect image correctly, please try again.</body>
                </html>''',

        "Angry":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Disgust":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Fear":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Happy":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Sad":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Surprise":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',

        "Neutral":'''<html>
          <body>
            <button type="button" onclick="redirect()">Click Me!</button>
            </body>
            <script type="text/javascript">
                function redirect(){
                    window.location.href="http://localhost:5003/music"
                }
            </script>
            </html>''',
        }

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def prediction(img):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    img = img.reshape((1,1,48,48))
    data = json.dumps({'data':img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')
    result = session.run([output_name],{input_name:data})
    prediction = torch.argmax(F.softmax(torch.from_numpy(np.array(result[0].squeeze())),dim=0),dim=0)
    return prediction


def predict(img):
    predicted_emotion = None
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
        )
    for (x,y,w,h) in faces:
        print("came here")
        roi_image = img[y:y+h,x:x+w]
        gray = cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.resize(gray,(48,48))
        img_rgb = np.array(img_rgb).astype(np.float32)
        img_rgb = np.expand_dims(img_rgb,axis=0)
        prediction_index = prediction(img_rgb)
        predicted_emotion = emotion[prediction_index]
    if predicted_emotion is None:
        return html["empty"]
    else:
        return html[predicted_emotion]


def launch():
    webcam = gd.inputs.Image(shape=(200,200),source="webcam")
    output = gd.outputs.HTML()
    gd.Interface(fn=predict,inputs=webcam,outputs=output,css="main.css").launch()

if __name__ == "__main__":
    launch()
