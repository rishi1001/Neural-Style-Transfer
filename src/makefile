epochs = 60
training=true
model=vgg
content_layers = conv_4
style_layers = 'conv_1 conv_2 conv_3 conv_4 conv_5'
# style_layers = 'conv_5'
folder = 2
show_img = false
style_weight = 1
run:
	for folder in 1 2 3 4 5 6; do \
		python3 main.py --folder $$folder --epochs $(epochs) --model $(model) --content_layers $(content_layers) --style_layers $(style_layers) --show_img $(show_img) --style_weight $(style_weight); \
	done 
clear_jpg:
	find ../style_weight -name \*.jpg -type f -delete