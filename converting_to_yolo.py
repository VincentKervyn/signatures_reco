import shutil
import os, sys, random
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os


# Setting the model parameters
shutil.copyfile('tobacco_data.yaml', 'yolov5/tobacco_data.yaml') # copying the custom_dataset.yaml file to the project repo
# setting number of classes to two (since the tobacco 800 dataset contains 2 classes, Logo & Signature)
