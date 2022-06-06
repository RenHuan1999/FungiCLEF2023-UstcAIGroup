# fgvc9_fungiclef

## steps

**train**

```shell
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py --cfg ./configs/MetaFG_meta_0_384.yaml --batch-size 64 --tag ${EXP_TAG} --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 64 --warmup-epochs 1 --dataset fungi --pretrain ./pretrained_model/metafg_0_inat21_384.pth --accumulation-steps 4 --num-workers 16 --opts DATA.IMG_SIZE 384
```

**test**

```shell
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12344  main.py --eval --cfg ./configs/MetaFG_meta_0_384.yaml --dataset fungi_test --resume output/MetaFG_meta_0/${EXP_TAG}/latest.pth --batch-size 64 --tag ${EXP_TAG}_test --opts DATA.IMG_SIZE 384
```

**ensamble and post process**

After runing `test`, we will get result{0-rank}.pkl which indicate the output of a single model, we can average ensamble the model outputs and do post process by runing `python post_avg.py`

## results

| team       | score   |
| :--------: | :----------: |
|xiong (ours)|**0.80426**|
|base|0.79759|
|USTC-IAT- United|0.79059|

**Our code are  based on [metaformer](https://github.com/dqshuai/MetaFormer)**
