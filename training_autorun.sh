#!/bin/bash

preprocessing_debug_mode=1
behavior_characterization_debug_mode=1
anomaly_detection_debug_mode=1
mode=training
anomalous_dataset_type=A
n_monitor_layers=3
preprocessing_test_percentage=0.5
normalize=Y
final_compression=N
final_compression_components=128
ad_technique=HC
behavior_characterization_validation_percentage=0.5

rm -f Input/Training/BC/Data/*
rm -f Input/Training/AD/Data/AN/*
rm -f Input/Training/AD/Data/TST/*
rm -f Input/Training/AD/Model/AE/*
rm -f Input/Training/AD/Model/HC/*
rm -f Input/Training/AD/Model/SC/*
rm -f Output/Training/PP/Preprocessing/Channels_Compression/*
rm -f Output/Training/PP/Preprocessing/Channels_Normalization/*
rm -f Output/Training/PP/Preprocessing/Final_Compression/*
rm -f Output/Training/PP/Data/AN/*
rm -f Output/Training/PP/Data/TR/*
rm -f Output/Training/PP/Data/TST/*
rm -f Output/Training/BC/Model/AE/*
rm -f Output/Training/BC/Model/HC/*
rm -f Output/Training/BC/Model/SC/*
rm -f Output/Training/AD/Classifications/*

python3 preprocessing.py $preprocessing_debug_mode $mode $anomalous_dataset_type $n_monitor_layers $preprocessing_test_percentage $normalize $final_compression $final_compression_components

cp Output/Training/PP/Preprocessing/Channels_Compression/*  Input/Inference/PP/Preprocessing/Channels_Compression

cp Output/Training/PP/Preprocessing/Channels_Normalization/* Input/Inference/PP/Preprocessing/Channels_Normalization

cp Output/Training/PP/Preprocessing/Final_Compression/* Input/Inference/PP/Preprocessing/Final_Compression

cp Output/Training/PP/Data/TR/* Input/Training/BC/Data

cp Output/Training/PP/Data/AN/* Input/Training/AD/Data/AN

cp Output/Training/PP/Data/TST/* Input/Training/AD/Data/TST

python3 behavior_characterization.py $behavior_characterization_debug_mode $ad_technique $behavior_characterization_validation_percentage

cp Output/Training/BC/Model/AE/* Input/Training/AD/Model/AE

cp Output/Training/BC/Model/HC/* Input/Training/AD/Model/HC

cp Output/Training/BC/Model/SC/* Input/Training/AD/Model/SC

cp Output/Training/BC/Model/AE/* Input/Inference/AD/Model/AE

cp Output/Training/BC/Model/HC/* Input/Inference/AD/Model/HC

cp Output/Training/BC/Model/SC/* Input/Inference/AD/Model/SC

python3 anomaly_detection.py $anomaly_detection_debug_mode $mode $ad_technique
