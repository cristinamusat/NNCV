**Cristina Musat**
- Codalab username: Cristina (scur1352)
- Email: [c.musat@student.tue.nl](mailto:c.musat@student.tue.nl) 

This repository contains the training script (`train.py`) and the final model (`model.py`) used for the Cityscapes Challenge, for the Peak Performance and Efficiency benchmarks.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/cristinamusat/NNCV.git
cd NNCV
```
2. Install the dependencies listed in requirements.txt
```bash
pip install -r requirements.txt
```

3. Submit the job and train the model using the parameters from main.sh
```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

`main.sh`

```bash
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 500 \
    --lr 0.0001 \
    --num-workers 12 \
    --seed 42 \
    --experiment-id "deeplabv3-resnet50-training" \
```