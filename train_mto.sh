export http_proxy=http://9.21.0.122:11113
export https_proxy=http://9.21.0.122:11113
export OMP_NUM_THREADS=8

pssh -h /tmp/pssh.hosts pkill python3

accelerate launch train_mto.py --sd_type "sd-2.1" --data "coco-6m" --task "mto" --stage "yolo"

