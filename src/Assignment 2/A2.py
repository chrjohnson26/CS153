# Note: you should not need to add any more libraries to complete 
# this assignment. If you think you need another library, please
# check with your instructor first.
import cv2
import numpy as np

# Note: this function is provided for you, and it may be useful.
# You should understand how it works.
def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB image.
    """
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Note: this function is provided for you, and it may be useful.
# You should understand how it works.
def load_alpha(impath):
    """
    Loads an image with a transparency channel and returns an RGB image and a transparency channel as two separate objects.
    Input:
    - impath: a string specifying the target image location. This image should have a transparency channel.
    Returns:
    - content: an RGB image containing the input image content.
    - alpha: a single channel image containing the input's transparency map
    """

    img_raw = cv2.imread(impath, cv2.IMREAD_UNCHANGED)
    img = img_raw.copy()
    img[:,:,0] = img_raw[:,:,2]
    img[:,:,2] = img_raw[:,:,0]

    content = img[:,:,0:3]
    alpha = img[:,:,3]

    return content, alpha

#TODO: Complete this function
def place_object(bkg_img, element, elmask, location, height):
    """
    Places an element into a specified location of background image at a specified scale.
    Parameters:
    - bkg_img: an RGB image into which the element should be inserted
    - element: an RGB image containing the object be inserted
    - elmask: a transparency mask that corresponds to the element being inserted
    - location: a tuple in (x,y) format which specifies the horizontal and vertical position, respectively, in normalized image scale coordinates to which the centroid of the minimal bounding box of the element should be inserted
    - height: specifies the height of the minimal bounding box for the element in normalized image scale. The element's width should be scaled accordingly to maintain its aspect ratio

    Returns an RGB image containing the composite output. This should *not* modify the input images, but should instead create a new image, and
    should be in 8-bit integer format.
    """

    return img_out.astype('uint8')
