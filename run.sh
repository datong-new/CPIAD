#CUDA_VISIBLE_DEVICES=7 python ig_attack.py --save_image_dir images_attack_grad_advbaseline1
#CUDA_VISIBLE_DEVICES=7 python ig_attack.py --save_image_dir images_attack_grad


attack_type=grad
model=faster


for ((i=1; i<17; i=i+2))
do
	echo $i
        CUDA_VISIBLE_DEVICES=3 python ig_attack_grid.py --save_image_dir "images_$attack_type" --attack_type $attack_type --filter $i --model $model
done

exit




filter=1
#attack_type=grad_input
#CUDA_VISIBLE_DEVICES=1 python ig_attack_grid.py --save_image_dir "images__filter10_$attack_type" --attack_type $attack_type --filter $filter

#attack_type=random
#CUDA_VISIBLE_DEVICES=6 python ig_attack_grid.py --save_image_dir "images__filter10_$attack_type" --attack_type $attack_type --filter $filter
#
#attack_type=integrated_grad
#CUDA_VISIBLE_DEVICES=2 python ig_attack_grid.py --save_image_dir "images__multiplybaseline_filter12_$attack_type" --attack_type $attack_type --filter $filter
