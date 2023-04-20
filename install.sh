#!/bin/bash

# 시스템 패키지 설치
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Python 패키지 설치
pip install -r requirements.txt
