FROM python:3.8.5

COPY ./ ./app
WORKDIR ./app
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install  -r requirements.txt
CMD ["python3","./camera.py"]

