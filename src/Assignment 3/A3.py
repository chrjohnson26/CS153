# Note: You may find the imutils library useful for this assignment.
# You should not need to add any additional libraries to complete this
# assignment. If you think you need another library, please check with
# your instructor first.
import cv2
import numpy as np
import os
import imutils as im

#### Provided Functions ####
# These functions are all provided for you. You may use them as you find useful.

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

def minbbox(mask):
    """
    Takes in a binary mask and returns a minimal bounding box around
    the non-zero elements.
    Input:
    - mask: a mask image (should contain only 0 and non-zero elements)
    Returns:
    - bbox: a list of four points in the form: [min_x, min_y, max_x, max_y]
    """

    min_h = np.where(mask > 0)[0].min()
    max_h = np.where(mask > 0)[0].max()
    min_w = np.where(mask > 0)[1].min()
    max_w = np.where(mask > 0)[1].max()

    return [min_w, min_h, max_w, max_h]

# A helper function I wrote for my solution; could easily be structured differently.
def pad_to_size(elimg, mask, scene, location):
    """
    Adds padding around element image and mask so that the element is
    centered in location corresponding to target in scene.
    Input:
    - elimg: The element image (assume it is three channels,
        but no restrictions on float or int)
    - mask: A mask corresponding to elimg
    - scene: The scene to insert into
    - location: The location the element will be inserted (relative to the center of the element)
    Returns:
    - big_img: a version of the element image padded out to the same dimensions as the scene
    - big_mask: a version of the element mask padded out to the same dimensions as the scene
    """
    [elh, elw, c] = elimg.shape

    big_img = np.zeros(scene.shape, dtype=float)
    big_mask = np.zeros((scene.shape[0], scene.shape[1]))

    skip_top = 0
    if int(location[1] - elh / 2) < 0:
        skip_top = abs(int(location[1] - elh / 2))

    skip_bot = 0
    if int(location[1] + elh / 2) > scene.shape[0]:
        skip_bot = int(location[1] + elh / 2) - scene.shape[0]

    skip_left = 0
    if int(location[0] - elw / 2) < 0:
        skip_left = abs(int(location[0] - elw / 2))

    skip_right = 0
    if int(location[0] + elw / 2) > scene.shape[1]:
        skip_right = int(location[0] + elw / 2) - scene.shape[1]

    targ_h_min = max(int(location[1] - elh / 2), 0)
    targ_h_max = min(int(location[1] + elh / 2), scene.shape[0])
    targ_w_min = max(int(location[0] - elw / 2), 0)
    targ_w_max = min(int(location[0] + elw / 2), scene.shape[1])

    # Due to rounding errors, this is sometimes off by one, so we do a last check
    # of the skip_top and skip_left variables, adjusting as needed
    if targ_h_max - targ_h_min < elh-skip_bot - skip_top:
        skip_top+=1
    if targ_w_max - targ_w_min < elw - skip_right - skip_left:
        skip_left+=1

    big_img[targ_h_min:targ_h_max, targ_w_min:targ_w_max, :] = elimg[skip_top:elh-skip_bot, skip_left:elw-skip_right, :]
    big_mask[targ_h_min:targ_h_max, targ_w_min:targ_w_max] = mask[skip_top:elh-skip_bot, skip_left:elw-skip_right]

    return big_img, big_mask

##### Question 1 #####

def green_extract(impath):
    """
    Loads an image from a specified location and performs green screen extraction.
    Input:
    - impath: a string specifying the target image location.
    Returns:
    - elements: a list of RGB images containing minimally bounded objects.
    - masks: a list of binary masks (numpy 'int' type) corresponding to the elements above
    """

    # This code is provided to you from our code in class to get you started.
    # You may decide to only use it for some of the images if you define custom rules
    # for some of the more challenging ones.
    thresh = 50
    fact = 1.5

    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (h,w,c) = img.shape

    gs_mask = np.zeros((h,w), dtype=bool)

    gs_mask = img[:,:,1] < fact*img[:,:,0]
    gs_mask = np.logical_or(gs_mask, img[:,:,1] < fact*img[:,:,2])
    gs_mask = np.logical_or(gs_mask, img[:,:,1] < thresh)

    # TODO: Any mask improvement steps and handling of multiple objects

    return elements, masks

##### Question 2 #####

def affine_insert(scene, element, elmask, eldepth, location, height,
                  angle = 0, scene_depth = None, alpha = 1):
    """
    Inserts an element into a scene according to a number of parameterized controls.
    Input:
    - scene: An RGB image into which the element should be inserted. Note, the scene
        object itself should be unchanged; you should return a new image.
    - element: An RGB image containing content information for an element to be inserted
        into the scene.
    - elmask: A mask corresponding to the element that specifies what content should be
        inserted into the scene.
    - eldepth: A scalar value of depth specifying the depth at which the object should
        be inserted into the scene.
    - location: a tuple in (x,y) format which specifies the horizontal and vertical position,
        respectively, in normalized image scale coordinates to which the centroid of the
        minimal bounding box of the element should be inserted.
    - height: specifies the height of the minimal bounding box for the element in
        normalized image scale. The element's width should be scaled accordingly
        to maintain its aspect ratio.
    - angle: The clockwise rotational angle that should be applied to the object.
    - scene_depth: A depth map providing a depth value for all points in the scene.
        A value of None corresponds to all scene points defaulting to infinite depth.
    - alpha: A transparency value for the element being inserted, with alpha=1
        corresponding to fully opaque.
    Returns:
    - out_scene: a composite scene with the element inserted.
    - out_depth: an updated depth map for the scene containing the element depth
        in all locations where the element appears.
    """

    # Rotating image according to the hyperparameter, angle
    element = im.rotate_bound(element, angle)
    elmask = im.rotate_bound(elmask, angle)
        
    # Getting element and background shape info
    bkg_height, bkg_width, _ = scene.shape
    ele_height, ele_width, _ = element.shape

    # Scale the image
    aspect_ratio = ele_width/ele_height
    new_ele_height = int(bkg_height * height)
    new_ele_width = int(new_ele_height * aspect_ratio)

    # Resize the element and elmask
    new_element = cv2.resize(element, (new_ele_width, new_ele_height))
    new_elmask = cv2.resize(elmask, (new_ele_width, new_ele_height))

    # Computing the location
    insert_x = int(location[0] * bkg_width)
    insert_y = int(location[1] * bkg_height)

    # Computing the col/row
    left = insert_x - int(new_ele_width/2)
    right = left + new_ele_width
    bottom = insert_y - int(new_ele_height/2)
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
    composite_img = scene.copy()
  
    # linear combination of the content and background
    print(ele_x_start, ele_x_end, ele_y_start, ele_y_end)
    print(bkg_x_start, bkg_x_end, bkg_y_start, bkg_y_end)
    composite_img[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end] = new_element[ele_y_start:ele_y_end, ele_x_start:ele_x_end] * new_elmask[ele_y_start:ele_y_end, ele_x_start:ele_x_end] + scene[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end] * (1-new_elmask[ele_y_start:ele_y_end, ele_x_start:ele_x_end])
    out_scene = composite_img

    # Post processing out_scene based on depth data
    if eldepth != None:
        change_coords = scene_depth < eldepth        # Indices which the scene_depth is less than eldepth


    
    return out_scene, out_depth

##### Question 3 #####

def custom_compose():
    """
    This is a functionalized script for performing image composition using the
    tools developed in this assignment. Note that you are free to hardcode things
    here; this only needs to return one image.
    Returns:
    - out_scene: a composite scene.
    """


    return out_scene
    
