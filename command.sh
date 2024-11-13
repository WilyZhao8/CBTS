python main.py --data /ai/rootdir/zwy/datasets/ImageNet_ILSVRC2012 \
  --lr 0.1 -p 100 --epochs 90 \
  --arch resnet50 --use_norm True \
  --wd 5e-4 --cos True \
  --cl_views rand-rand --num_classes 1000  \
  --batch-size 256

# python main.py --data /ai/zwy/zwy/zwy/public_data/ImageNet_ILSVRC2012 \
#   --lr 0.1 -p 200 --epochs 90 \
#   --arch resnext50 --use_norm True \
#   --wd 5e-4 --cos True \
#   --cl_views rand-rand --num_classes 1000  \
#   --batch-size 236


# python main.py --data /ai/zwy/zwy/zwy/public_data/iNaturalist \
#   --lr 0.2 -p 200 --epochs 100 \
#   --dataset inat \
#   --arch resnet50 --use_norm True \
#   --wd 1e-4 --cos True \
#   --cl_views rand-rand --num_classes 8142  \
#   --batch-size 256


