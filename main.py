from tifffile import imread


def load_training_img(img_path):
    """Load training images
    :param img_path: path of images
    :return: data output format to be determined
    """
    img = imread(img_path)

    return img


test = load_training_img('train_images/train_images/0a0f8e20b1222b69416301444b117678.tiff')
print(test)
print(test.shape)
print(type(test))
