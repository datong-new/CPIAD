model=yolo
i=15

#attack_type=integrated_grad_ones
#CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
#
#
#attack_type=integrated_grad_zeros
#CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
#
#
#attack_type=integrated_grad_half
#CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
#
#attack_type=integrated_grad_rand
#CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
#

attack_type=integrated_grad_near
CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model


exit

attack_type=grad_input
CUDA_VISIBLE_DEVICES=2 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
attack_type=grad
CUDA_VISIBLE_DEVICES=2 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
attack_type=random
CUDA_VISIBLE_DEVICES=2 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model

#for ((i=1; i<15; i=i+2))
#do
#	echo $i
#        CUDA_VISIBLE_DEVICES=1 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
#done
