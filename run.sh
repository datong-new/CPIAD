#CUDA_VISIBLE_DEVICES=7 python ig_attack.py --save_image_dir images_attack_grad_advbaseline1
#CUDA_VISIBLE_DEVICES=7 python ig_attack.py --save_image_dir images_attack_grad

#attack_type=grad_input
#CUDA_VISIBLE_DEVICES=4 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type

#attack_type=grad
#CUDA_VISIBLE_DEVICES=5 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type

#attack_type=random
#CUDA_VISIBLE_DEVICES=6 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type
#
attack_type=integrated_grad
CUDA_VISIBLE_DEVICES=7 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type
