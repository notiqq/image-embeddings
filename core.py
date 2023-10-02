import os
import os.path
import cv2
from PIL import Image
from rembg import remove

def remove_background_from_images(index_directory, processed_directory):
    exists = os.path.exists(processed_directory)
    if not exists:
        os.makedirs(processed_directory)

    index_images = get_files_in_directory(index_directory)

    for file in index_images:
        input_path = build_path(index_directory, file)
        output_path = build_path(processed_directory, file)
        remove_background_from_image(input_path, output_path)
        cutoff_transparency_with_bounding_box(output_path,output_path)
        convert_png_to_jpg(output_path, output_path)

def build_path(directory, file):
    return os.path.join(directory, file)

def get_files_in_directory(directory):
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(directory+'/'+f)]
    return files


def remove_background_from_image(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


def cutoff_transparency_with_bounding_box(input_path, output_path):
    im = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    x, y, w, h = cv2.boundingRect(im[..., 3])
    im2 = im[y:y+h, x:x+w, :]
    cv2.imwrite(output_path, im2)

def convert_png_to_jpg(input_path, output_path):
    im = Image.open(input_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(output_path)