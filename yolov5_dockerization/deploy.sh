#!/bin/bash
sudo docker build -t object_detect_v1 .
sudo docker rm -f object_detect_v1
sudo docker run -d -it -p 5058:5058 --name object_detect_v1 object_detect_v1
