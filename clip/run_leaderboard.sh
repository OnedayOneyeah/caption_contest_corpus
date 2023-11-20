task=matching
lr=.000005
sp=5

accelerate launch clip/train_clip.py 5 $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12;
