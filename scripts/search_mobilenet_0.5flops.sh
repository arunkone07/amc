python3 amc_search.py \
    --job=train \
    --model=mobilenet \
    --dataset=imagenet \
    --reward=acc_flops_reward \
    --data_root=/srv/datasets/ \
    --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
    --seed=2018 
    # --preserve_ratio=0.5 \
    # --lbound=0.2 \
    # --rbound=1 \
