#!/bin/bash

cmd="python main.py --model=EMT_DLFR --augmentation=feat_random_drop --dataset=MOSI --eval-noise-type=rawa_bg_park"
echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
eval $cmd