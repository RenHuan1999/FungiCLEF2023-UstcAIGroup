# MetaFG_meta_2_384
EXP_TAG="MetaFG_meta_2_384_bs18_epoch64_poison_trainval"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12346  main.py --cfg ./configs/MetaFG_meta_2_384.yaml --batch-size 18 --tag ${EXP_TAG} --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 64 --warmup-epochs 1 --dataset fungi --pretrain ./pretrained_model/metafg_2_inat21_384.pth --accumulation-steps 4 --num-workers 18 --opts DATA.IMG_SIZE 384
