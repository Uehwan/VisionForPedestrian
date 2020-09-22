#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv vision-ped
echo "Activating virtual environment"

source $PWD/vision-ped/bin/activate

$PWD/vision-ped/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
$PWD/vision-ped/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/vision-ped/bin/pip install -r requirements.txt
