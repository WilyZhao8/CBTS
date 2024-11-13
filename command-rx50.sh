python main.py --data /ai/zwy/zwy/zwy/public_data/ImageNet_ILSVRC2012 \
  --lr 0.1 -p 200 --epochs 90 \
  --arch resnext50 --use_norm True \
  --wd 5e-4 --cos True \
  --cl_views rand-rand --num_classes 1000  \
  --batch-size 240




