task=matching
lr=.000005
dataset_path='/data/mjjung/The_new_yorker_caption_contest' # define your dataset path.

#for sp in 1 2 3 4
#do
#  accelerate launch --config_file clip/my_config_file.yaml \
#  clip/train_clip.py $sp $task \
#  --warmup 200 \
#  --clip_model ViT-L/14@336px \
#  --pad 1 \
#  --lr $lr \
#  --use_accelerate 1 \
#  --batch_size 16 \
#  --n_epochs 12 \
#  --dataset_path $dataset_path \
#  ${@:1}
#done

for sp in 1
do
#  CUDA_VISIBLE_DEVICES=0,1,2,3
  accelerate launch --config_file clip/my_config_file.yaml \
  clip/train_clip.py $sp $task \
  --warmup 200 \
  --clip_model ViT-L/14@336px \
  --pad 1 \
  --lr $lr \
  --use_accelerate 1 \
  --batch_size 16 \
  --n_epochs 12 \
  --dataset_path $dataset_path \
  ${@:1}
done