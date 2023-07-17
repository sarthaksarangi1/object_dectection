FROM python:3.7-slim

ADD ./app /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

WORKDIR app

CMD [ "python", "detect.py" ]
