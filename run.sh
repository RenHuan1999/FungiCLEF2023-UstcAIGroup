# convnext large
EXP_TAG="convnext_large_exp"
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg ./configs/convnext_large.yaml --batch-size 24 --tag ${EXP_TAG} --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 80 --warmup-epochs 1 --dataset fungi --pretrain ./pretrained_model/convnext_large_22k_1k_384.pth --accumulation-steps 6 --num-workers 32 --opts DATA.IMG_SIZE 384
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12344  main.py --eval --cfg ./configs/convnext_large.yaml --dataset fungi_test --resume output/convnext_large/${EXP_TAG}/latest.pth --batch-size 6 --tag ${EXP_TAG}_test --opts DATA.IMG_SIZE 384
