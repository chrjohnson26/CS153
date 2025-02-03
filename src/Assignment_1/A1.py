import cv2
import numpy as np

### Question 1
def load_img(impath):
    """
    Loads an image from a specified location and returns it in RGB format.
    Input:
    - impath: a string specifying the target image location.
    Returns an RGB image.
    """
    img = cv2.imread(impath) # Using imread function to read image
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


## Question 2

# TODO: Complete this function
def slice_video(basepath, start_num, frames, numpix = 1, offset = 0):    
    """
    Generates a static image by taking a slice through a video.
    Input:
    - basepath: a string specifying the folder where the video frames are stored and the start of the file name.
    - start_num: the starting number of the frame to use.
    - frames: an argument specifying the number of frames to process. This will be a number greater than 0 and less than or equal to the number of available frames for a given video.
    - numpix: an argument specifying how many pixels to sample from each frame. Note, this could be a non-integer value, but will be greater than 0. 
    - offset: An optional argument that specifies an offset from the left edge to start sampling frames. Any pixels to the left of this value will be filled in from the first frame. This will be an integer greater than or equal to 0.
    Return:
    - comp_img: an RGB image containing the composite image created by sampling columns from each of the video frames.
    """
    img = load_img(basepath + str(start_num) + '.png')

    comp_img = np.zeros(img.shape) # initialized as an empty black image

    [_,w,_] = img.shape

    numpix = int(numpix) # flooring numpix

    # TODO: input error check for numpix and offset
    img_mask = img[:, :offset] # specifying the pixels due to the offset value
    comp_img[:, :offset] = img_mask

    for fidx in range(frames):
        # TODO: input error check for numpix and offset
        cur_frame = start_num + fidx
        img = load_img(basepath + str(cur_frame) + '.png') # loads cur_frame to img
        img_mask = img[:, offset + numpix*fidx: offset + numpix*fidx + numpix]
        comp_img[:, offset + numpix*fidx: offset + numpix*fidx + numpix] = img_mask

    img = load_img(basepath + str(start_num + frames-1) + '.png')
    img_mask = img[:, offset + numpix*frames:]
    comp_img[:, offset+numpix*frames:] = img_mask

    return comp_img.astype('uint8')