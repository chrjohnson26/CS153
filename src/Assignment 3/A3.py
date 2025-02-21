# Note: You may find the imutils library useful for this assignment.
# You should not need to add any additional libraries to complete this
# assignment. If you think you need another library, please check with
# your instructor first.
import cv2
import numpy as np
import os
import imutils as im
from scipy.io import loadmat

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
    alpha = 0
    beta = 0
    if impath == 'imgs/test_imgs/ball_toss.png':
        alpha = 100
    elif impath == 'imgs/test_imgs/calden_umbrella.png':
        thresh = 60
        fact = 1.6
        alpha = 30
    elif impath == 'imgs/test_imgs/head.png' or impath == 'imgs/test_imgs/francine_poppins.png':
        beta = 40
        alpha = 40
    elif impath == 'imgs/custom_imgs/crouch.jpg':
        beta = 50
        alpha = 40


    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (h,w,c) = img.shape

    # gs_mask = np.zeros((h,w), dtype=bool)

    gs_mask = img[:,:,1] < beta+fact*img[:,:,0]
    gs_mask = np.logical_or(gs_mask, img[:,:,1] < alpha+fact*img[:,:,2])
    gs_mask = np.logical_or(gs_mask, img[:,:,1] < thresh)

    # min_w, min_h, max_w, max_h=minbbox(gs_mask)

    # Parameter to control the morpholgy element size (from lecture demo slides)
    morph_size = 8

    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
    eromask = cv2.erode(gs_mask.astype(np.uint8), morph_kern, iterations=1) # using cv2.erode to remove residuals
    gs_mask = cv2.dilate(eromask, morph_kern, iterations=1) # using cv2 dialate function to enlarge the people


    # Finding the connected components within the mask
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(gs_mask.astype('uint8'), connectivity=8)
    # reformatting stats output
    indexes = stats[:,4].argsort()

    component_indices = indexes[-retval:]   # Taking only the (# of retval) components
    retained_component_indicies = [component_indices[i] for i in range(retval-1)] # 
    labels[np.invert(np.isin(labels, retained_component_indicies))] = 0
    binary_masks = [(labels == i+1).astype('uint8') for i in range(retval-1)]

    masks = [mask[minbbox(mask)[1]:minbbox(mask)[3], minbbox(mask)[0]:minbbox(mask)[2]] for mask in binary_masks]
    elements = [img[minbbox(mask)[1]:minbbox(mask)[3], minbbox(mask)[0]:minbbox(mask)[2]] for mask in binary_masks]
    
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
    elmask = im.rotate_bound(elmask.astype('float'), angle)
        
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
    out_scene = scene.copy()
    
    # Checking scene_depth
    if scene_depth is None:
        scene_depth = np.full(scene.shape[:2], np.inf)

    out_depth = scene_depth.copy()

    # Specifying the element reigon and scene region
    ele_region = new_element[ele_y_start:ele_y_end, ele_x_start:ele_x_end]
    mask_region = new_elmask[ele_y_start:ele_y_end, ele_x_start:ele_x_end]
    scene_region = out_scene[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]
    scene_depth_reigon = scene_depth[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]

    # Creating the composite image
    bin_depth_mask = eldepth < scene_depth_reigon # bin_depth_mask is True where element should show up and False where background should show up
    mask_region = mask_region[:,:,None] * bin_depth_mask[:,:,None]
    ele_masked = ele_region * mask_region * alpha
    scene_masked = scene_region * (1-mask_region)*alpha
    out_scene[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]= ele_masked + scene_masked 

    # Updating the depth map to only represent the insertion element
    depth_mask = eldepth * mask_region
    depth_mask[depth_mask == 0] = np.inf # setting the 0 depth pixels in the mask to infinity so they dont get detected in the min function
    out_depth[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end] = np.minimum(out_depth[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end], depth_mask[:,:,0])
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
    bgname = '_1295'

    d = loadmat('imgs/depth_maps/depth' + bgname + '.mat')
    depth_map = d['dimg']
    bckgrd = load_img('imgs/backgrounds/image' + bgname + '.png')

    


    elements, masks= green_extract('imgs/custom_imgs/josh_isaac.jpg')
    elements1, masks1 = green_extract('imgs/custom_imgs/crouch.jpg')
    crouch_im = elements1[0]
    crouch_mask = masks1[0]
    ji = elements[0]
    ji_mask = masks[0]

    scene, scene_depth = affine_insert(bckgrd, ji, ji_mask, 2.3, [0.1,0.2], 0.4, 90, depth_map)
    scene2, scene_depth2 = affine_insert(scene, crouch_im, crouch_mask, 3.2, [0.65,0.40], 0.4, 10, scene_depth)
    out_scene, _ = affine_insert(scene2, crouch_im, crouch_mask, 1.5, [0.5,0.6], 0.4, -30, scene_depth2)


    return out_scene
    
