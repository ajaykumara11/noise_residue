# the cython code
import sys
sys.path.append('.')
import numpy
cimport numpy
import cython
cimport cython

# TO COMPILE THIS:
# python3.9 Paper3_Work/setup.py build_ext --inplace

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def calcStdDev_iterative(numpy.ndarray[double, ndim=2] block, int as, int useMean,int start_i,int end_i,int start_j,int end_j,int startx,int endx,int starty,int endy):

    cdef int no_exclPixels = 1, i, j
    cdef double stdDev_local
    cdef numpy.ndarray[double, ndim=1] results = numpy.zeros([2])
    cdef int max_iterations = 50
    cdef int iteration_count = 0
    #no_exclPixels = 1
    while(no_exclPixels > 0):
        
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"maximum interation reached without conv for {start_i},{end_i},{start_j},{end_j},{startx},{endx},{starty},{endy}")
            
        mean_local = numpy.nanmean(block)
        stdDev_local = numpy.nanstd(block)
        #print("stdDev_local", mean_local, stdDev_local)
        mask = ~numpy.isnan(block)
        #print("Pixels:", block[mask].shape[0])
        #print("useMean:", useMean)
        no_exclPixels = 0

        '''
        Having a look at speeding it up by removing the nested for loop
        dataBlock = block[i,j]
        AbsdataBlock = np.abs(dataBlock)
        dataBlock[stdDev_local*3 > AbsdataBlock]=numpy.nan'''
        # If max iterations exceeded, mark convergence failure
        
            
        for i in range(0, block.shape[0]):
            for j in range(0, block.shape[1]):
                if(useMean==1):
                    if(block[i,j] > mean_local + 3*stdDev_local):
                        block[i,j] = numpy.nan
                        no_exclPixels = no_exclPixels+1
                        # print(f"pixel excluded: {no_exclPixels}")
                    if(block[i,j] < mean_local - 3*stdDev_local):
                        block[i,j] = numpy.nan
                        no_exclPixels = no_exclPixels+1
                        # print(f"pixel excluded: {no_exclPixels}")
                else:
                    if(block[i,j] > 3*stdDev_local):
                        block[i,j] = numpy.nan
                        no_exclPixels = no_exclPixels+1
                    if(block[i,j] < -3*stdDev_local):
                        block[i,j] = numpy.nan
                        no_exclPixels = no_exclPixels+1
        #print("no_exclPixels:", no_exclPixels)
    
    results[0] = stdDev_local
    results[1] = mean_local
    print(f"done for:{start_i},{end_i},{start_j},{end_j},{startx},{endx},{starty},{endy} with iteration of conv {iteration_count}")
    return results
    


import os
def movingWindowStdDev(numpy.ndarray[double, ndim=2] data, 
                       numpy.ndarray[double, ndim=2] mu, int sizeWindow, int abs, 
                       int useMean, int startx, int endx, int starty, int endy):
    #print(startx, endx, starty, endy)

    cdef int i, j
    cdef int start_i, start_j, end_i, end_j
    cdef numpy.ndarray[double, ndim=3] stdDevArray = numpy.zeros([2, endx-startx, endy-starty])
    #print("In function")

    for i in range(startx, endx):
        # if(i%10==0):
        #     print("i:",i)
        for j in range(starty, endy):
            #just pick pixels from inside the disk
            # print("indices",i,j,startx,endx,starty,endy)
            if(numpy.isnan(mu[i,j]) != True):
                #print(data.shape)
                start_i = i-sizeWindow
                end_i = i+sizeWindow
                start_j = j-sizeWindow
                end_j = j+sizeWindow
                if (i-sizeWindow < 0):
                    start_i = 0
                if (i+sizeWindow > data.shape[0]):
                    end_i = data.shape[0]
                if (j-sizeWindow < 0):
                    start_j = 0
                if (j+sizeWindow > data.shape[0]):
                    end_j = data.shape[0]
                currentBlock = data[start_i:end_i, start_j:end_j]
                stdDev, mean_local = calcStdDev_iterative(currentBlock.copy(), abs, useMean,start_i,end_i, start_j,end_j,startx,endx,starty,endy)
                stdDevArray[0, i-startx, j-starty] = stdDev
                stdDevArray[1, i-startx, j-starty] = mean_local

    return stdDevArray

# the pooling code
import numpy as np
import matplotlib.pyplot as plt
# Load the normalized image data from the .npy file
loaded_data = np.load('/home/kumara/Yeo 2013/normalised_image_data.npy', allow_pickle=True)
first_normalized_image = loaded_data[0]

# Displaying the shape of the first normalized image
print(f'Shape of the first normalized image: {first_normalized_image.shape}')
image_plot=plt.imshow(first_normalized_image,cmap='gray',vmin=-10,vmax=10)
bar=plt.colorbar(image_plot)
plt.show()

array=first_normalized_image
mu=array.copy()
signed=1
useMean=1
sizeWindow=200
size = array.shape[0]
print(size)

indices = [[0, int(size/2), 0, int(size/4)],
        [0, int(size/2), int(size/2), 3*int(size/4)],
        [0, int(size/2), int(size/4), int(size/2)],
        [0, int(size/2), 3*int(size/4), size],
        [int(size/2), size, 0, int(size/4)],
        [int(size/2), size, int(size/4), int(size/2)],
        [int(size/2), size, int(size/2), 3*int(size/4)],
        [int(size/2), size, 3*int(size/4), size]]

import multiprocessing as mp
import sys
sys.path.append('/home/kumara/Yeo 2013/')

from kinga_cython import movingWindowStdDev

pool = mp.Pool(8)

result_Std = pool.starmap(movingWindowStdDev, [(array, 
                            mu, sizeWindow, signed, useMean, *f) for f in indices])


results = np.asarray(result_Std)

#
results_array = np.zeros((2, array.shape[0], array.shape[1]))


for i in range(0,len(indices)):
    startx = indices[i][0]
    endx = indices[i][1]
    starty = indices[i][2]
    endy = indices[i][3]
    results_array[:, startx:endx, starty:endy]=results[i,:,:,:]     

pool.close()  
pool.join()  

print("Done.")
