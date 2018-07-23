
import os

from PIL import Image, ImageChops
import numpy as np
#input_dir = "/usr/src/lego_classification/part_recognition/lego_images/crop_testing"
#input_dir = "/usr/src/lego_classification/part_recognition/lego_images/lego_images_copy"
input_dir = "/usr/src/lego_images_cropped"

def resize_square(path, size):
	im = Image.open(path)
	if im.mode == 'RGBA':
		print("Mode:" + im.mode)
		
		#im = im.convert('RBG')
		#im.save(image.toString() + ".jpg")
		
		im.load() # required for png.split()

		background = Image.new("RGB", im.size, (255, 255, 255))
		background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
		im = background

	if im.mode == 'L':

		rgbim = Image.new("RGB", im.size)
		rgbim.paste(im)
		im = rgbim
		
	im = make_square(im)
	im = im.resize((size, size), Image.ANTIALIAS)
	im.save(path[:-4]+ ".jpg")


def listdir_nohidden(path):
	#Returns a list without hidden files
	files = os.listdir(path)
	for f in files:
		if f.startswith('.'):
			files.remove(f)
	return files

def test(file):
	im = Image.open(file)
	#im = crop_white(im)
	im = make_square(im)
	im.save(file)

def make_square(im):

	fill_color = (255, 255, 255, 0)

	x, y = im.size
	size = max(x, y)
	new_im = Image.new('RGB', (size, size), fill_color)
	new_im.paste(im, ((size - x) / 2, (size - y) / 2))
	return new_im

