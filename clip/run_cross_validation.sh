#for task in matching
#  do for sp in 1 2 3 4
#    do for lr in .00001 .00005 .000005;
#      do accelerate launch clip/train_clip.py $sp $task --warmup 200 --clip_model ViT-L/14@336px \
#      --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12;
#    done;
#  done;
#done;

#for task in matching
#  do for sp in 1
#    do for lr in .000005
#      do accelerate launch clip/train_clip.py $sp $task --warmup 200 --clip_model ViT-L/14@336px \
#      --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12 --debug;
#    done;
#  done;
#done;

task=matching
lr=.000005
for sp in 1 2 3 4
do
  accelerate launch clip/train_clip.py $sp $task \
  --warmup 200 \
  --clip_model ViT-L/14@336px \
  --pad 1 \
  --lr $lr \
  --use_accelerate 1 \
  --batch_size 16 \
  --n_epochs 12 \
  ${@:1}
done

#sp=5
#accelerate launch clip/train_clip.py 5 $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12;
