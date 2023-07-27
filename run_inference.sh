# MetaFG_meta_2_384
EXP_TAG="MetaFG_meta_2_384_bs18_epoch64_poison_trainval"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12346  main.py --eval --cfg ./configs/MetaFG_meta_2_384.yaml --dataset fungi_test --resume output/MetaFG_meta_2/${EXP_TAG}/latest.pth --batch-size 6 --tag ${EXP_TAG}_epoch64_test --opts DATA.IMG_SIZE 384
