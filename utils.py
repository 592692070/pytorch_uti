import sys
import os
import numpy as np
import torch

def setgpu(gpuinput):
    print('using gpu '+gpuinput)
    os.environ['CUDA_VISIBLE_DEVICES']=gpuinput
    return len(gpuinput.split(','))




# if __name__ =='__main__':
#     setgpu('1,2,3')
#     exit()