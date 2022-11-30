epochs = 1000
training=true
model=googlenet
# content_layers = conv_4 		# vgg
# content_layers = sequential_2			# resnet
content_layers = 'other_3'			# googlenet
# style_layers = 'conv_1 conv_2 conv_3 conv_4 conv_5'  # vgg
# style_layers = 'sequential_1 sequential_2 sequential_3'		# resnet
style_layers = 'other_2 other_3 other_4 other_5'  # googlenet
folder = 6
show_img = false
# style_weight = 10000000
style_weight = 1000000
run:						# change this for more folders
	for folder in 1 2 3 4 5 6; do \
		python3 main.py --folder $$folder --epochs $(epochs) --model $(model) --content_layers $(content_layers) --style_layers $(style_layers) --show_img $(show_img) --style_weight $(style_weight); \
	done 
clear_jpg:
	find ../start_random -name \*.jpg -type f -delete