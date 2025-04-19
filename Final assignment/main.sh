wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 500 \
    --lr 0.0001 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "deeplabv3-resnet50-training" \