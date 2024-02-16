#!/bin/bash


while getopts 'm:e:c:t:l:w:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) cps_w=$OPTARG;;
    esac
done
echo $method
echo $cuda

epoch=300
echo $epoch

labeled_data="labeled_40p"
unlabeled_data="unlabeled_40p"
folder="Task_"${task}"_40p/"
cps="AB"

echo $folder


python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold1 --seed 549 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r #--start_mix 50
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold1 -g ${cuda} --cps ${cps}

python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold2 --seed 521 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r #--start_mix 50
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold2 -g ${cuda} --cps ${cps}
#python code/evaluate_Ntimes.py --exp ${folder}${method}${exp} --folds 2 --cps ${cps}
#END_COMMENT
python code/train_${method}.py --task ${task} --exp ${folder}${method}${exp}/fold3 --seed 999 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r #--start_mix 50
python code/test.py --task ${task} --exp ${folder}${method}${exp}/fold3 -g ${cuda} --cps ${cps}

python code/evaluate_Ntimes.py --task ${task} --exp ${folder}${method}${exp} --folds 3 --cps ${cps}