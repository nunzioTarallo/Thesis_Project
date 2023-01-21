#!/bin/bash

echo $PYTHONPATH
export PYTHONPATH=/home/nunzio/caffe/python

ripetition=2
for ((counter=$ripetition; counter>0; counter--))
do
start=$(date +%s.%N)

preprocessing_debug_mode=1
behavior_characterization_debug_mode=1
anomaly_detection_debug_mode=1
mode=inference
anomalous_dataset_type=A
n_monitor_layers=3
preprocessing_test_percentage=0.5
normalize=Y
final_compression=N
final_compression_components=128
ad_technique=AE
behavior_characterization_validation_percentage=0.5
numero_monitor=3
tipo_dataset=A

rm -f Input/Inference/PP/Data/*
rm -f Output/Inference/PP/Data/*
rm -f Input/Inference/AD/Data/*

python3 cnn_inference.py $numero_monitor $tipo_dataset

python3 preprocessing.py $preprocessing_debug_mode $mode $anomalous_dataset_type $n_monitor_layers $preprocessing_test_percentage $normalize $final_compression $final_compression_components

cp Output/Inference/PP/Data/*  Input/Inference/AD/Data

python3 anomaly_detection.py $anomaly_detection_debug_mode $mode $ad_technique

duration=$(echo "$(date +%s.%N) - $start " | bc)
LC_NUMERIC="en_US.UTF-8" execution_time= printf "%.2f\n" $duration >>tempi.csv


done
