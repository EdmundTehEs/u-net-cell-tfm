
import numpy as np                                             # basic math
import cv2                                                     # for optical flow
import skimage.io as io                                        # reading in images
import pandas as pd                                            # for parameter file
import glob as glob                                            # for finding files
import os                                                      # for making directory to store displacement files


def TFM_optical_flow(cell_path='.', pyr_scale=0.25, levels=4, winsize=24,
                     iterations=4, poly_n=7, poly_sigma=1.25,
                     save_files=True, return_data=False):
    """Calculate bead displacements using Farneback optical flow.

    Parameters
    ----------
    cell_path : str, optional
        Folder containing a registered bead stack and reference image.
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma : float or int
        Parameters passed to ``cv2.calcOpticalFlowFarneback``.

    save_files : bool, optional
        If ``True`` (default) write displacement images and parameter CSVs to disk.
    return_data : bool, optional
        If ``True`` return the displacement fields as arrays.

    Side Effects
    ------------
    When ``save_files`` is ``True`` saves displacement images in ``displacement_files/``.
    """
    current_dir = os.getcwd()
    os.chdir(cell_path)
    # Create a dictionary of our PIV parameters
    PIV_params = {
    "method" : 'Farneback Optical Flow',
    "pyr_scale" : pyr_scale,
    "levels" : levels,
    "winsize" : winsize,
    "iterations" : iterations,
    "poly_n" : poly_n,
    "poly_sigma" : poly_sigma
    }
    
    # Convert the dictionary to a DataFrame
    PIV_params_df = pd.DataFrame(PIV_params, index=[0])
    if save_files:
        # Write the parameters to a CSV file for saving
        PIV_params_df.to_csv('PIV_params.csv')

    # Run a for loop with all the images

    # read in reference image
    ref_file_list = glob.glob('*_reference.tif')
    if len(ref_file_list) == 0:
        raise FileNotFoundError("No '*_reference.tif' image found")
    if not os.path.isfile(ref_file_list[0]):
        raise FileNotFoundError(f"Expected reference image '{ref_file_list[0]}' not found")
    reference_image = io.imread(ref_file_list[0])

    # make a directory to store all the displacement files
    if save_files and os.path.isdir('displacement_files/') == False:
        os.mkdir('displacement_files/')

    # read in image stack
    stack_file = ref_file_list[0][:-14] + '_registered.tif'
    if not os.path.isfile(stack_file):
        raise FileNotFoundError(f"Registered bead stack '{stack_file}' not found")
    image_stack = io.imread(stack_file)

    # correct the stack shape if there's only one image
    if len(image_stack.shape) == 2:
        temp = np.zeros((1,image_stack.shape[0],image_stack.shape[1]))
        temp[0] = image_stack.copy()
        image_stack = temp.copy()

    # Get the number of images in the stack
    N_images = image_stack.shape[0]

    disp_u = np.zeros((N_images, reference_image.shape[0], reference_image.shape[1]))
    disp_v = np.zeros_like(disp_u)

    for t, image_stack_plane in enumerate(image_stack):
        
        # perform optical flow forward and backward
        flow_forward = cv2.calcOpticalFlowFarneback(reference_image, image_stack_plane, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
        flow_reverse = cv2.calcOpticalFlowFarneback(image_stack_plane, reference_image, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
        
        # take the average of the two flow fields
        flow = (flow_forward - flow_reverse) / 2

        disp_u[t] = flow[:,:,0]
        disp_v[t] = flow[:,:,1]

        if save_files:
            # save the displacements as images
            io.imsave('displacement_files/disp_u_%03d.tif' % (t), flow[:,:,0], check_contrast=False)
            io.imsave('displacement_files/disp_v_%03d.tif' % (t), flow[:,:,1], check_contrast=False)


    os.chdir(current_dir)
    if return_data:
        return disp_u, disp_v
    return
