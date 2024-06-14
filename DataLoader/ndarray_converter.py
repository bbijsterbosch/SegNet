import numpy as np
import pandas as pd

def convert_cvs_ndarray(path_to_file, path_to_save, print_bool=False):
    #convert into pd frame
    class_dict = pd.read_csv(path_to_file)
    # extract the values 
    rgb_values = class_dict[['r', 'g', 'b']].to_numpy()
    np.save(path_to_save, rgb_values)

    if print_bool==True:
        print(rgb_values, "\n")
        print(f"The path to the file where the classes per pixel value are stored is: {path_to_file}","\n")
        print(f"The path to the file where the classes per pixel value in numpy format are stored is: {path_to_save}", "\n")
    
    return rgb_values


path_to_file = "/home/jzwanen/computer_vision/SegNet/CamVidDataSet/CamVid/class_dict.csv"
path_to_save = "/home/jzwanen/computer_vision/SegNet/CamVidDataSet/CamVid/seg_classes"
classes =  convert_cvs_ndarray(path_to_file, path_to_save, True)




