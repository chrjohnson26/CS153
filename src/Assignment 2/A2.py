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
    
    # get background, content, mask dimensions
    bkg_height, bkg_width, _ = bkg_img.shape
    ele_height, ele_width, _ = element.shape

    # get background, content, mask dimensions
    bkg_height, bkg_width, _ = bkg_img.shape
    ele_height, ele_width, _ = element.shape

    # get ratio (width to height) and new element/mask dimensions
    print(element.shape)
    aspect_ratio = ele_width/ele_height
    new_ele_height = int(bkg_height*height)
    new_ele_width =  int(new_ele_height * aspect_ratio)

    # resize element and mask based on the new dimensions
    new_element = cv2.resize(element, (new_ele_width, new_ele_height))
    new_elmask = cv2.resize(elmask,  (new_ele_width, new_ele_height))

    # convert mask to three channels of identical values so we can compute the composite
    new_elmask = cv2.cvtColor(new_elmask.astype('uint8'), cv2.COLOR_GRAY2RGB)
    new_elmask = new_elmask / 255. 

    # Calculate the location of element based on location
    scaled_x = int(location[0] * bkg_width) 
    scaled_y = int(location[1] * bkg_height)



    # Getting the rows and cols we are imposing element/mask on the background
    left = scaled_x - int(new_ele_width/2)
    right = left + new_ele_width
    bottom = scaled_y - int(new_ele_height/2)
    top = bottom + new_ele_height 

    # Calculating the element bounds
    ele_x_start = max(0, -left)
    ele_y_start = max(0 , -bottom)
    ele_x_end   = min(new_ele_width, bkg_width - left)
    ele_y_end   = min(new_ele_height, bkg_height - bottom)

    # Calculating the background bounds where the element will go
    bkg_x_start = max(0, left)
    bkg_y_start = max(0, bottom)
    bkg_x_end = min(bkg_width, right)
    bkg_y_end = min(bkg_height, top)

    # initialize composite image as the background
    composite_img = bkg_img.copy()
  
    # linear combination of the content and background
    # composite_img[bottom:top, left:right] = new_element * new_elmask + bkg_img[bottom:top, left:right] * (1-new_elmask)
    print(ele_x_start, ele_x_end, ele_y_start, ele_y_end)
    print(bkg_x_start, bkg_x_end, bkg_y_start, bkg_y_end)
    composite_img[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end] = new_element[ele_y_start:ele_y_end, ele_x_start:ele_x_end] * new_elmask[ele_y_start:ele_y_end, ele_x_start:ele_x_end] + bkg_img[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end] * (1-new_elmask[ele_y_start:ele_y_end, ele_x_start:ele_x_end])
    img_out = composite_img

    return img_out.astype('uint8')
