from flask import Flask,render_template,url_for
import os
import gradio as gr

app = Flask(__name__)
metadata ={
    "angry":{
        "emotion":"angry",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "sad":{
        "emotion":"sad",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "neutral":{
        "emotion":"neutral",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "happy":{
        "emotion":"happy",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "surprise":{
        "emotion":"surprise",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "fear":{
        "emotion":"fear",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you.",
        },
    "disgust":{
        "emotion":"disgust",
        "title":"Anger Songs",
        "quote":"Your anger gives you great power. But if you let it, it will destroy you."
        }


    }
        

@app.route("/music")
def music():
    songList=[]
    emotion="disgust"
    for _,_,files in os.walk("static/songs/"+emotion):
        songList.append(files)
        print(files)
    return render_template("main.html",data=metadata[emotion],songs=songList[0])

@app.route("/test")
def test():
    return render_template('camera.html')
    
if __name__ == "__main__":
    app.run(debug=True,port=5003)
