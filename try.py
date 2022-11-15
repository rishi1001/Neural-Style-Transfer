import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--content_layers',type=str,nargs='+',default='conv_4', help='content layers to use')
args = parser.parse_args()

print(args.content_layers)