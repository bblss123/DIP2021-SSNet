python main.py --dataset SHHB \
--model SSNet \
--train-files .\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\train\\train.txt \
--val-files .\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\train\\val.txt \
--gpu-devices 1 \
--lr 1e-5 \
--optim adam \
--loss bceloss \
--checkpoints ./checkpoints \
--summary-writer ./runs/demo