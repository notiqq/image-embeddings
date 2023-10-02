import sys
import os
import config
import core
import models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from getopt import getopt

similarity_model = models.SimilaritySearch()


def show_images(full_images_paths):
    fig = plt.figure(figsize=(8, 8))
    columns = len(full_images_paths)
    rows = 1
    for i in range(1, columns*rows +1):
        img = mpimg.imread(full_images_paths[i-1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

def clean_up_directory(directory):
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

def parse_path(input_path):
    filename = os.path.basename(input_path).split('/')[-1]
    directory = os.path.dirname(input_path)
    return filename, directory

def main(arguments):
    try:
        image_path, param_name, count = arguments
        count = int(count)
    except Exception as ex:
        print("wrong arguments, required format: python3 inference.py image.jpg --top_n 3")
        return

    filename, directory = parse_path(image_path)
    exists = os.path.exists(config.INFERENCE_OUTPUTH_PATH)
    if not exists:
        os.makedirs(config.INFERENCE_OUTPUTH_PATH)
    output_image_path = core.build_path(config.INFERENCE_OUTPUTH_PATH,filename)
    core.remove_background_from_image(image_path, output_image_path)
    core.cutoff_transparency_with_bounding_box(output_image_path,output_image_path)
    core.convert_png_to_jpg(output_image_path, output_image_path)
    search_results = similarity_model.search(config.INFERENCE_OUTPUTH_PATH, filename, count)
    print(search_results)
    images_paths_to_show = [image_path]
    for result in search_results:
        images_paths_to_show.append(core.build_path(config.INDEX_DIRECTORY,result[1]))
    
    show_images(images_paths_to_show)
    clean_up_directory(config.INFERENCE_OUTPUTH_PATH)

if __name__ == "__main__":
   main(sys.argv[1:])