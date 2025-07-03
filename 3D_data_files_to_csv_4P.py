#This is in effect the same code as before, but now in 3D!

#This is the new code that will process all the CSVs of points into twos CSVs for R
# One will have fish summary stats, while the other will have the fish comparison values
#For the fish one the columns will be:
# Year, Month, Day, Trial, Abalation, Darkness, Flow, Fish, Tailbeat Num, Heading, Speed, TB Frequency

# For the between fish comparisons the columns will be: 
# Year, Month, Day, Trial, Abalation, Darkness, Flow, Fishes, Tailbeat Num, X Distance, Y Distance, Distance, Angle, Heading Diff, Speed Diff, Synchonization

#This is a lot of columns. But now instead of having multiple .npy files this will create an object for each of the positional data
# CSVs and then add them all together in the end. This will ideally make things easier to graph for testing, and not require so many 
# nested for loops. Fish may be their own objects inside of the trial objects so that they can be quickly compared. Which may mean that I need to 
# take apart fish_core_4P.py. In the end I think a lot of this will be easier to do with pandas and objects instead of reading line by line.

from scipy.signal import hilbert, savgol_filter, medfilt, butter, lfilter, sosfilt
from scipy.interpolate import splprep, splev, interp2d
from scipy.spatial import ConvexHull

from scipy.spatial import distance_matrix
import networkx as nx
import pylab

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec 
import pandas as pd
import numpy as np
import random
import math
import os, sys

species = ["Giant Danio","Cherry Barb","Neon Tetra"][2]

# fish_dates = {"2024_04_02": "Cherry Barb",
#               "2024_02_21": "Cherry Barb"}

species_mean_len = {"Giant Danio":0.083197,
                    "Cherry Barb":0.038,
                    "Neon Tetra":0.032875}

species_sd_len = {"Giant Danio":0.62998,
                  "Cherry Barb":0.0047,
                  "Neon Tetra":0.0023}

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

fps = 60

#The moving average window is more of a guess tbh
moving_average_n = 35

#Tailbeat len is the median of all frame distances between tailbeats
tailbeat_len = 19

#Fish len is the median of all fish lengths in pixels
#Scale is different becasue of calibration
fish_len = species_mean_len[species]

#Used to try and remove weird times where fish extend
# Fish SD
fish_sd = species_sd_len[species]

# Fish Len Max?
#Okay but now we need it in BL instead
fish_max_len = (fish_len + 3*fish_sd) / fish_len

#Header list for reading the raw location CSVs
header = list(range(4))

def calc_mag_vec(p1):
    return math.sqrt((p1[0])**2 + (p1[1])**2)

def calc_mag(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_dist_np_2D(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist

def get_dist_np_3D(x1s,y1s,z1s,x2s,y2s,z2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2+(z1s-z2s)**2)
    return dist

def moving_average(x, w):
    #Here I am using rolling instead of convolve in order to not have massive gaps from a single nan
    return  pd.Series(x).rolling(window=w, min_periods=1).mean()

def normalize_signal(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    divisor = max(max_val,abs(min_val))

    return data/divisor

def mean_tailbeat_chunk(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.nanmean(data[start:end])

    return mean_data[::tailbeat_len]

def median_tailbeat_chunk(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.nanmedian(data[start:end])

    return mean_data[::tailbeat_len]

def angular_mean_tailbeat_chunk(data,tailbeat_len):
    #data = np.deg2rad(data)

    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        data_range = data[start:end]

        cos_mean = np.nanmean(np.cos(data_range))
        sin_mean = np.nanmean(np.sin(data_range))

        #SIN then COSINE
        #angular_mean = np.rad2deg(np.arctan2(sin_mean,cos_mean))
        angular_mean = np.arctan2(sin_mean,cos_mean)
        mean_data[k] = angular_mean

    return mean_data[::tailbeat_len]

def mean_tailbeat_chunk_sync(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.nanmean(data[start:end])

    return np.power(2,abs(mean_data[::tailbeat_len])*-1)

def x_intercept(x1,y1,x2,y2):
    m = (y2-y1)/(x2-x1)
    intercept = (-1*y1)/m + x1

    return intercept

#Calculates the uniformity of a distribution of phases or angles as a dimensionless number from 0 to 1
#Data must be given already normalized between 0 and 2pi

def rayleigh_cor(data):
    #Make an empy array the length of long axis of data
    out_cor = np.zeros(data.shape[0])

    #Get each time point as an array
    for i,d in enumerate(data):
        #Calcualte the x and y coordinates on the unit circle from the angle
        xs = np.cos(d)
        ys = np.sin(d)

        #Take the mean of x and of y
        mean_xs = np.nanmean(xs)
        mean_ys = np.nanmean(ys)

        #Find the magnitude of this new vector
        magnitude = np.sqrt(mean_xs**2 + mean_ys**2)

        out_cor[i] = magnitude

    return out_cor

def where_dupes(x_data,y_data,z_data):
    all_diff = np.abs(np.diff(x_data)) + np.abs(np.diff(y_data)) + np.abs(np.diff(z_data))

    output = [True]
    diff_true = all_diff > 0
    output.extend(diff_true)

    not_nans = ~np.isnan(x_data)

    output = output & not_nans

    return output

def add_back_nans(new_data,valid_points,shape):
    output = np.zeros(shape)
    output[output == 0] = np.nan
    output[valid_points] = new_data

    return output

def array_shortener(array, min_len, max_len):
    
    array_len = len(array)
    start = (array_len - min_len) // 2
    stop = array_len - math.ceil((array_len - min_len) / 2)

    return array[start:stop]

class fish_data:
    def __init__(self, name, data, scorer, flow):
        #This sets up all of the datapoints that I will need from this fish
        self.name = name
        self.head_x = data[scorer][name]["head"]["x"].to_numpy() / fish_len
        self.head_y = data[scorer][name]["head"]["y"].to_numpy() / fish_len
        self.head_z = data[scorer][name]["head"]["z"].to_numpy() / fish_len

        self.midline_x = data[scorer][name]["midline2"]["x"].to_numpy() / fish_len
        self.midline_y = data[scorer][name]["midline2"]["y"].to_numpy() / fish_len
        self.midline_z = data[scorer][name]["midline2"]["z"].to_numpy() / fish_len

        self.vec_x = []
        self.vec_y = []
        self.vec_z = []
        self.vec_xy = []

        self.flow = flow

        #These are the summary stats for the fish 
        self.yaw_heading = []
        self.pitch_heading = []
        self.yaw_heading_ps = []
        self.speed = []
        self.accel = []

        #Okay, so I want to remove data where fish are too long
        # So I am going to just do that, and void it out here with nans
        # Since all further functions draw from these positional values, I just null them here
        self.remove_OOB_fish()

        self.smooth_points_spline()

        #This calcualtes the summary stats
        self.calc_yaw_heading()
        self.calc_pitch_heading()
        self.calc_speed()
        self.calc_accel()
        self.calc_yaw_heading_ps()

        self.calc_tb_freq()

        self.speed_sd = np.nanstd(self.speed)
        self.speed_COV =  self.speed_sd / np.nanmean(self.speed)

        #self.graph_values()        

    #Remove fish that are out of bounds of the space.
    # Fish are about 2 inches long, filming area is about 25 x 10 x 10, so 12 x 5 x 5 BL,
    # If they are out of that +- 2 BL remove them
    def remove_OOB_fish(self):
        min_x = -1
        max_x = 0.6 / fish_len + 1

        min_y = -1
        max_y = 0.25 / fish_len + 1

        min_z = -1
        max_z = 0.25 / fish_len + 1

        is_OOB = (self.head_x < min_x) + (self.head_x > max_x) + (self.head_y < min_y) + (self.head_y > max_y) + (self.head_z < min_z) + (self.head_z > max_z)

        self.head_x[is_OOB] = np.nan
        self.head_y[is_OOB] = np.nan
        self.head_z[is_OOB] = np.nan

        self.midline_x[is_OOB] = np.nan
        self.midline_y[is_OOB] = np.nan
        self.midline_z[is_OOB] = np.nan

    def smooth_points(self):

        smoothing_window = tailbeat_len//3

        self.head_x = moving_average(self.head_x,smoothing_window)
        self.head_y = moving_average(self.head_y,smoothing_window)
        self.head_z = moving_average(self.head_z,smoothing_window)

        self.midline_x = moving_average(self.midline_x,smoothing_window)
        self.midline_y = moving_average(self.midline_y,smoothing_window)
        self.midline_z = moving_average(self.midline_z,smoothing_window)

    def smooth_points_spline(self):

        self.head_x_raw = self.head_x
        self.head_y_raw = self.head_y
        self.head_z_raw = self.head_z

        self.midline_x_raw = self.midline_x
        self.midline_y_raw = self.midline_y
        self.midline_z_raw = self.midline_z

        smoothing_factor = 0.15

        head_valid_points = where_dupes(self.head_x_raw,self.head_y_raw,self.head_z_raw)

        head_x_raw_nonan = self.head_x_raw[head_valid_points]
        head_y_raw_nonan = self.head_y_raw[head_valid_points]
        head_z_raw_nonan = self.head_z_raw[head_valid_points]

        if np.isnan(head_x_raw_nonan).all():
            print("NO POINTS")
            return

        tck, u = splprep([head_x_raw_nonan,head_y_raw_nonan,head_z_raw_nonan], s = smoothing_factor, k = 5)
        spline_head_points = splev(u, tck)    

        self.head_x = add_back_nans(spline_head_points[0],head_valid_points,self.head_x_raw.shape)
        self.head_y = add_back_nans(spline_head_points[1],head_valid_points,self.head_y_raw.shape)
        self.head_z = add_back_nans(spline_head_points[2],head_valid_points,self.head_z_raw.shape)

        if np.isnan(self.head_x).all():
            print("\n ERROR WITH SPLINE!!! \n")

        midline_valid_points = where_dupes(self.midline_x_raw,self.midline_y_raw,self.midline_z_raw)

        midline_x_raw_nonan = self.midline_x_raw[midline_valid_points]
        midline_y_raw_nonan = self.midline_y_raw[midline_valid_points]
        midline_z_raw_nonan = self.midline_z_raw[midline_valid_points]

        tck, u = splprep([midline_x_raw_nonan,midline_y_raw_nonan,midline_z_raw_nonan], s = smoothing_factor, k = 5)
        spline_head_points = splev(u, tck)         

        self.midline_x = add_back_nans(spline_head_points[0],midline_valid_points,self.midline_x_raw.shape)
        self.midline_y = add_back_nans(spline_head_points[1],midline_valid_points,self.midline_y_raw.shape)
        self.midline_z = add_back_nans(spline_head_points[2],midline_valid_points,self.midline_z_raw.shape)


    #This function calcualtes the yaw heading of the fish at each timepoint
    #We are using body heading now, so midline to head, not head to next head
    def calc_yaw_heading(self):
        #Then we create a vector of the head minus the midline 
        self.vec_x = self.head_x - self.midline_x
        self.vec_y = self.head_y - self.midline_y

        #Then we use arctan to calculate the heading based on the x and y point vectors
        #Becasue of roll we don't want to the last value since it will be wrong
        self.yaw_heading = np.arctan2(self.vec_y,self.vec_x)

        # print(self.vec_x)
        # print(self.vec_y)
        # print(self.yaw_heading)

        # sys.exit()

    #This function calcualtes the pitch heading of the fish at each timepoint
    #We are using body heading now, so midline to head, not head to next head
    def calc_pitch_heading(self):
        #Then we create a vector of the head minus the midline 
        self.vec_x = self.head_x - self.midline_x
        self.vec_y = self.head_y - self.midline_y
        self.vec_z = self.head_z - self.midline_z

        self.vec_xy = get_dist_np_2D(0,0,self.vec_x,self.vec_y)

        #Then we use arctan to calculate the heading based on the x and y point vectors
        #Becasue of roll we don't want to the last value since it will be wrong
        self.pitch_heading = np.arctan2(self.vec_xy,self.vec_z)

    #calcualtes the yaw heading change per second
    def calc_yaw_heading_ps(self):

        chunked_yaw_mean = angular_mean_tailbeat_chunk(self.yaw_heading, tailbeat_len)

        chunked_yaw_heading_vec = np.column_stack((np.cos(chunked_yaw_mean), np.sin(chunked_yaw_mean)))
        
        dot_prods = np.zeros(len(chunked_yaw_heading_vec)) + np.nan
        cross_prods = np.zeros(len(chunked_yaw_heading_vec)) + np.nan

        for t in range(1,len(chunked_yaw_heading_vec)-1):

            chunked_yaw_heading_vec_prev = chunked_yaw_heading_vec[t-1]
            chunked_yaw_heading_vec_next = chunked_yaw_heading_vec[t+1]
            chunked_yaw_heading_vec_prev_perp = [-1*chunked_yaw_heading_vec[t-1][1],chunked_yaw_heading_vec[t-1][0]]

            dot_prods[t] = np.dot(chunked_yaw_heading_vec_prev,chunked_yaw_heading_vec_next)
            cross_prods[t] = np.dot(chunked_yaw_heading_vec_next,chunked_yaw_heading_vec_prev_perp)

        self.yaw_heading_ps = np.arccos(dot_prods) * np.sign(cross_prods) * fps/(2*tailbeat_len)
        self.yaw_heading_ps = self.yaw_heading_ps[1:-1]

    def calc_speed(self):
        #First we get the next points on the fish
        head_x_next = np.roll(self.head_x, -1)
        head_y_next = np.roll(self.head_y, -1)
        head_z_next = np.roll(self.head_z, -1)

        head_x_prev = np.roll(self.head_x, 1)
        head_y_prev = np.roll(self.head_y, 1)
        head_z_prev = np.roll(self.head_z, 1)

        #Then we create a vector of the future point minus the last one
        speed_vec_x = head_x_next - head_x_prev
        speed_vec_y = head_y_next - head_y_prev
        speed_vec_z = head_z_next - head_z_prev

        #Then we add the flow to the x value
        #Since (0,0) is in the upper left a positive vec_x value value means it is moving downstream
        #so I should subtract the flow value 
        #The flow value is mutliplied by the fish length since the vec_x values are in pixels, but it is in BLS so divide by fps
        vec_x_flow = speed_vec_x - ((self.flow)/fps)

        #It is divided in order to get it in body lengths and then times fps to get BL/s
        self.speed = np.sqrt(vec_x_flow**2+speed_vec_y**2+speed_vec_z**2) * fps / 2

        self.speed = self.speed[1:-1]

    def calc_accel(self):

        chunked_speed = mean_tailbeat_chunk_sync(self.speed,tailbeat_len)

        #remove the edge cases where rolling makes invalid values
        self.accel = (np.roll(chunked_speed, -1) - np.roll(chunked_speed, 1))[1:-1] * fps/(2*tailbeat_len)

    def calc_tb_freq(self):
        #This code is written to get a sense of the fishes taibeat based on the movement of the midline around the head to head heading
        #We gon't have the tailtip data, so we're taking the oscilations we can get

        #First get the total number of frames
        total_frames = len(self.head_x)

        out_tailtip_perps = []

        #My old code does this frame by frame. There may be a way to vectorize it, but I'm not sure about that yet
        for i in range(total_frames-1):
            #Create a vector from the head to the midline and from the head to the next head
            tailtip_vec = np.asarray([self.head_x[i]-self.midline_x[i],self.head_y[i]-self.midline_y[i],self.head_z[i]-self.midline_z[i]])
            midline_vec = np.asarray([self.head_x[i]-self.head_x[i+1],self.head_y[i]-self.head_y[i+1],self.head_z[i]-self.head_z[i+1]])

            #Then we make the midline vector a unit vector
            vecDist = np.sqrt(midline_vec[0]**2 + midline_vec[1]**2 + midline_vec[2]**2)
            midline_unit_vec = midline_vec/vecDist

            #We take the cross product of the midline unit vecotr to get a vector perpendicular to it
            perp_midline_vector = np.cross(midline_unit_vec,[0,0,1])

            #Finally, we calcualte the dot product between the vector perpendicular to midline vector and the 
            # vector from the head to the tailtip in order to find the perpendicular distance from the midline
            # to the tailtip
            out_tailtip_perps.append(np.dot(tailtip_vec,perp_midline_vector))

        self.tailtip_perp = moving_average(normalize_signal(out_tailtip_perps),moving_average_n)
        
        tailtip_perp_no_nan = self.tailtip_perp[~np.isnan(self.tailtip_perp)]

        if len(tailtip_perp_no_nan) > 0:

            self.hilbert_trans = np.unwrap(np.angle(hilbert(tailtip_perp_no_nan)))

            hilbert_phase = np.gradient(self.hilbert_trans)

            phase_nan = np.zeros(self.tailtip_perp.shape)
            phase_nan[phase_nan == 0] = np.nan
            phase_nan[~np.isnan(self.tailtip_perp)] = hilbert_phase

            self.instant_freq = np.abs(phase_nan / (2 * np.pi) * fps)

        else:
            self.hilbert_trans = np.zeros(self.tailtip_perp.shape) + np.nan
            self.instant_freq = np.zeros(self.tailtip_perp.shape) + np.nan

    #Thsi function allows me to graph values for any fish without trying to cram it into a for loop somewhere
    def graph_values(self):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(ncols = 3, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])
        ax0.plot(self.head_x, self.head_y)
        ax0.scatter(self.head_x[0], self.head_y[0])
        ax0.plot(self.head_x_raw, self.head_y_raw)
        ax0.set_title("Fish Path (Blue = Head, Orange = Midline)")

        ax2 = plt.subplot(gs[2,1])
        ax2.plot(range(len(self.tailtip_perp)), self.tailtip_perp)
        ax2.set_title("Tailbeat Perp")

        ax3 = plt.subplot(gs[0,1])
        ax3.plot(range(len(self.speed)), self.speed)
        ax3.set_title("Speed")

        ax4 = plt.subplot(gs[1,1])
        ax4.plot(range(len(self.yaw_heading)), self.yaw_heading)
        ax4.set_title("Heading")

        ax5 = plt.subplot(gs[0,2])
        ax5.plot(range(len(self.hilbert_trans)), self.hilbert_trans)
        ax5.set_title("Hilbert")

        ax6 = plt.subplot(gs[1,2])
        ax6.plot(range(len(self.instant_freq)), self.instant_freq)
        ax6.set_title("Hilbert Frq")

        plt.show()

class fish_comp:
    def __init__(self, fish1, fish2):
        self.name = fish1.name + "x" + fish2.name
        self.f1 = fish1
        self.f2 = fish2

        self.x_diff = []
        self.y_diff = []
        self.z_diff = []
        self.dist = []
        self.angle = []
        self.yaw_heading_diff = []
        self.pitch_heading_diff = []
        self.speed_diff = []

        self.calc_dist()
        self.calc_angle()
        self.calc_yaw_heading_diff()
        self.calc_pitch_heading_diff()
        self.calc_speed_diff()

        self.calc_relative_x()

        #self.graph_values()

    def calc_dist(self):        
        #Divided to get it into bodylengths
        self.x_diff = (self.f1.head_x - self.f2.head_x)
        #the y_diff is negated so it faces correctly upstream
        self.y_diff = -1*(self.f1.head_y - self.f2.head_y)
        self.z_diff = (self.f1.head_z - self.f2.head_z)

        self.dist = get_dist_np_3D(0,0,0,self.x_diff,self.y_diff,self.z_diff)

    def calc_angle(self):
        #Calculate the angle of the x and y difference in degrees
        #angle_diff = np.rad2deg(np.arctan2(self.y_diff,self.x_diff))

        #This makes it from 0 to 360
        #angle_diff_360 = np.mod(abs(angle_diff-360),360)
        #This rotates it so that 0 is at the top and 180 is below the fish for a sideways swimming fish model
        #self.angle = np.mod(angle_diff_360+90,360)

        #12/1/21: Back to making change notes. Now keeping it as the raw -180 to 180

        #3/7/24 Doing this with dot products to not run into issues with axes

        f1_head = np.column_stack((self.f1.head_x, self.f1.head_y))
        f1_midline = np.column_stack((self.f1.midline_x, self.f1.midline_y))
        f2_head = np.column_stack((self.f2.head_x, self.f2.head_y))

        dot_prods = np.zeros(len(self.f1.head_x)) + np.nan
        cross_prods = np.zeros(len(self.f1.head_x)) + np.nan

        for t in range(len(self.f1.head_x)):

            unit_f1_body_vec = (f1_head[t] - f1_midline[t]) / calc_mag(f1_head[t],f1_midline[t])
            unit_f1_f2_head_vec  = (f2_head[t] - f1_head[t]) / calc_mag(f1_head[t],f2_head[t])
            unit_f1_body_vec_perp = [-1*unit_f1_body_vec[1],unit_f1_body_vec[0]]

            dot_prods[t] = np.dot(unit_f1_body_vec,unit_f1_f2_head_vec)
            cross_prods[t] = np.dot(unit_f1_f2_head_vec,unit_f1_body_vec_perp)

        self.angle = np.arccos(dot_prods) * np.sign(cross_prods)

        #sys.exit()

    #Now with a dot product!
    def calc_yaw_heading_diff(self):
        f1_vector = np.asarray([self.f1.vec_x,self.f1.vec_y]).transpose()
        #print(f1_vector)
        f2_vector = np.asarray([self.f2.vec_x,self.f2.vec_y]).transpose()

        self.yaw_heading_diff = np.zeros(len(self.f1.vec_x))

        for i in range(len(self.f1.vec_x)):
            dot_product = np.dot(f1_vector[i], f2_vector[i])

            prod_of_norms = np.linalg.norm(f1_vector[i]) * np.linalg.norm(f2_vector[i])
            self.yaw_heading_diff[i] = np.arccos(dot_product / prod_of_norms)

    #Now with a dot product!
    def calc_pitch_heading_diff(self):
        f1_vector = np.asarray([self.f1.vec_xy,self.f1.vec_z]).transpose()
        #print(f1_vector)
        f2_vector = np.asarray([self.f2.vec_xy,self.f2.vec_z]).transpose()

        self.pitch_heading_diff = np.zeros(len(self.f1.vec_xy))

        for i in range(len(self.f1.vec_x)):
            dot_product = np.dot(f1_vector[i], f2_vector[i])

            prod_of_norms = np.linalg.norm(f1_vector[i]) * np.linalg.norm(f2_vector[i])
            self.pitch_heading_diff[i] = np.arccos(dot_product / prod_of_norms)

    def calc_heading_diff_filtered(self):
        #Makes sure that head wiggle doesn't mess up polarization
        f1_heading_og = self.f1.heading
        f2_heading_og = self.f2.heading

        self.f1.heading = savgol_filter(self.f1.heading,tailbeat_len,1)
        self.f2.heading = savgol_filter(self.f2.heading,tailbeat_len,1)

        self.heading_diff = np.arctan2(np.sin(self.f1.heading-self.f2.heading),
                                       np.cos(self.f1.heading-self.f2.heading))

        for i in range(len(self.f1.heading)):
            print(f1_heading_og[i],f2_heading_og[i],self.f1.heading[i],self.f2.heading[i],self.heading_diff[i])

    def calc_speed_diff(self):
        self.speed_diff = self.f1.speed - self.f2.speed

    def calc_relative_x(self):
        f1_head = np.column_stack((self.f1.head_x, self.f1.head_y))
        f1_midline = np.column_stack((self.f1.midline_x, self.f1.midline_y))
        f2_head = np.column_stack((self.f2.head_x, self.f2.head_y))

        self.relative_x = np.zeros(len(self.f1.head_x)) + np.nan

        for t in range(len(self.f1.head_x)):
            unit_f1_body_vec = (f1_head[t] - f1_midline[t]) / calc_mag(f1_head[t],f1_midline[t])
            f1_f2_head_vec  = (f2_head[t] - f1_head[t])

            self.relative_x[t] = np.dot(f1_f2_head_vec,unit_f1_body_vec)

    def graph_values(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(ncols = 5, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])

        ax0.scatter(self.f1.head_x, self.f1.head_y, alpha = np.linspace(0,1,num = len(self.f1.head_x)), s = 2)
        ax0.scatter(self.f2.head_x, self.f2.head_y, alpha = np.linspace(0,1,num = len(self.f2.head_x)), s = 2)
        ax0.set_title("Fish Path (Blue = Fish 1, Orange = Fish 2)")

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(range(len(self.dist)), self.dist)
        ax1.set_title("Distance")

        ax2 = plt.subplot(gs[1,1])
        ax2.plot(range(len(self.angle)), self.angle)
        ax2.set_title("Angle")

        ax3 = plt.subplot(gs[0,2])
        ax3.plot(range(len(self.f1.speed)), self.f1.speed)
        ax3.set_title("Fish 1 Speed")

        ax4 = plt.subplot(gs[1,2])
        ax4.plot(range(len(self.f2.speed)), self.f2.speed)
        ax4.set_title("Fish 2 Speed")

        ax5 = plt.subplot(gs[2,2])
        ax5.plot(range(len(self.speed_diff)), self.speed_diff)
        ax5.set_title("Speed Difference")

        ax6 = plt.subplot(gs[0,3])
        ax6.plot(range(len(self.f1.yaw_heading)), self.f1.yaw_heading)
        ax6.set_title("Fish 1 Heading")

        ax7 = plt.subplot(gs[1,3])
        ax7.plot(range(len(self.f2.yaw_heading)), self.f2.yaw_heading)
        ax7.set_title("Fish 2 Heading")

        ax8 = plt.subplot(gs[2,3])
        ax8.plot(range(len(self.yaw_heading_diff)), self.yaw_heading_diff)
        ax8.set_title("Heading Difference")

        plt.show()

class school_comps:
    def __init__(self, fishes, n_fish, flow):
        self.fishes = fishes
        self.n_fish = n_fish
        self.flow = flow

        self.school_center_x = []
        self.school_center_y = []
        self.school_center_z = []
        self.school_x_sd = []
        self.school_y_sd = []
        self.school_z_sd = []

        self.group_speed = []
        self.polarization = []

        self.correlation_strength = []
        self.nearest_neighbor_distance = []

        self.school_areas = []
        self.school_groups = []

        self.school_height = []
        self.school_tailbeat_freq = []

        self.calc_school_pos_stats()

        self.remove_and_smooth_points()

        self.calc_school_speed()
        self.calc_school_polarization()
        self.calc_nnd()
        self.calc_school_area()

        self.calc_school_groups_all_points()

        self.calc_school_height()

        self.calc_school_tailbeat_freq()

        #self.graph_values()

    def calc_school_pos_stats(self):
        school_xs = [fish.head_x for fish in self.fishes]
        school_ys = [fish.head_y for fish in self.fishes]
        school_zs = [fish.head_z for fish in self.fishes]

        self.school_center_x = np.nanmean(school_xs, axis=0)
        self.school_center_y = np.nanmean(school_ys, axis=0)
        self.school_center_z = np.nanmean(school_zs, axis=0)

        self.school_x_sd = np.nanstd(school_xs, axis=0)
        self.school_y_sd = np.nanstd(school_ys, axis=0)
        self.school_z_sd = np.nanstd(school_zs, axis=0)

    def remove_and_smooth_points(self):
        threshold = 0.01

        self.school_center_x = savgol_filter(self.school_center_x,31,1)
        self.school_center_y = savgol_filter(self.school_center_y,31,1)
        self.school_center_z = savgol_filter(self.school_center_z,31,1)

    def calc_school_speed(self):
        # Changing this to mean speed of the fish in the school

        all_fish_speeds = [fish.speed for fish in self.fishes]

        self.group_speed = np.nanmean(all_fish_speeds, axis = 0)

    def calc_school_polarization(self):
        #formula from McKee 2020
        sin_headings = np.sin([fish.yaw_heading for fish in self.fishes])
        cos_headings = np.cos([fish.yaw_heading for fish in self.fishes])

        self.polarization = (1/self.n_fish)*np.sqrt(np.nansum(sin_headings, axis=0)**2 + np.nansum(cos_headings, axis=0)**2)

    def calc_nnd(self):
        #first we make an array to fill with the NNDs 
        nnd_array  = np.zeros((len(self.school_center_x),self.n_fish,self.n_fish)) + np.nan

        #now calculate all nnds
        for i in range(self.n_fish):
            for j in range(self.n_fish):

                if i != j:
                    fish1 = self.fishes[i]
                    fish2 = self.fishes[j]

                    #dists = get_dist_np_2D(fish1.head_x,fish1.head_y,fish2.head_x,fish2.head_y)

                    dists = get_dist_np_3D(fish1.head_x,fish1.head_y,fish1.head_z,fish2.head_x,fish2.head_y,fish2.head_z)

                    for t in range(len(self.school_center_x)):
                        nnd_array[t][i][j] = dists[t]

        #Then we get the mins of each row (or column, they are the same), and then get the mean for the mean
        # NND for that timepoint

        min_array = np.nanmin(nnd_array,axis = 1)

        #We then NAN out all timepoints where we don't have at least 4 fish to make up the school at that time
        num_nans = np.count_nonzero(np.isnan(min_array), axis=1).astype(np.float) 
        num_nans[num_nans > 4] = np.nan
        num_nans[num_nans <= 4] = 1

        #Multiply by fishlen and then by 100 to get it in cm not BL or meters
        self.nearest_neighbor_distance = np.nanmean(np.nanmin(nnd_array,axis = 1), axis = 1)*num_nans #*fish_len*100
        
    def calc_school_area(self):
        school_xs = np.asarray([fish.head_x for fish in self.fishes])
        school_ys = np.asarray([fish.head_y for fish in self.fishes])
        school_zs = np.asarray([fish.head_z for fish in self.fishes])

        self.school_areas = [np.nan for i in range(len(school_xs[0]))]

        for i in range(len(school_xs[0])):
            x_row = school_xs[:,i]
            y_row = school_ys[:,i]
            z_row = school_zs[:,i]

            mask = ~np.isnan(x_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(y_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(z_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            if len(x_row) >= 4:
                points = np.column_stack((x_row,y_row,z_row))

                hull = ConvexHull(points)

                self.school_areas[i] = hull.volume**2

    def calc_school_groups_all_points(self):
        min_BL_for_groups = 2

        school_xs = np.asarray([fish.head_x for fish in self.fishes])

        #Get all the fish head and tailtip points
        school_heads = np.asarray([[fish.head_x for fish in self.fishes],[fish.head_y for fish in self.fishes],[fish.head_z for fish in self.fishes]])
        school_midlines = np.asarray([[fish.midline_x for fish in self.fishes],[fish.midline_y for fish in self.fishes],[fish.midline_z for fish in self.fishes]])
    
        #Set up the final array to be filled in
        self.school_groups = [np.nan for i in range(len(school_xs[0]))]

        for i in range(len(school_xs[0])):

            #Get just the points for the current frame
            head_points = np.asarray([item for item in zip(school_heads[0][:,i], school_heads[1][:,i], school_heads[2][:,i])])
            midline_points = np.asarray([item for item in zip(school_midlines[0][:,i], school_midlines[1][:,i], school_midlines[2][:,i])])

            #Remove NANs from head and tailtip so they aren't added as nodes later
            mask = ~np.isnan(head_points) & ~np.isnan(midline_points)

            #Reshape to make them fit and remove NANs with mask
            head_points = head_points[mask]
            head_points = head_points.reshape((int(len(head_points)/3), 3))

            midline_points = midline_points[mask]
            midline_points = midline_points.reshape((int(len(midline_points)/3), 3))

            #Save them in an arrayto go over
            point_types = [head_points,midline_points]

            dm_array = []

            #Get head vs all other points
            for p_other in point_types:
                dm_array.append(distance_matrix(head_points,p_other))

            #Turn into an array
            dm_array = np.asarray(dm_array)
            #print(dm_array)

            #Get the shortest distance combo of the four
            dm_min = np.nanmin(dm_array, axis = 0)
            #print(dm_min)

            #Divide by fish length
            dm_min = dm_min

            #Find where it is less than the set BL for grouping
            dm_min_bl = dm_min <= min_BL_for_groups

            #Make into a graph and then get the number of points.
            G = nx.from_numpy_array(dm_min_bl)

            n_groups = len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])

            self.school_groups[i] = n_groups

            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, with_labels=True)
            # plt.show()

            # sys.exit()

    def calc_school_height(self):
        school_zs = np.asarray([fish.head_z for fish in self.fishes])

        self.school_height = np.nanmax(school_zs, axis = 0) - np.nanmin(school_zs, axis = 0)

    def calc_school_tailbeat_freq(self):
        school_freqs = np.asarray([fish.instant_freq for fish in self.fishes])

        self.school_tailbeat_freq = np.nanmean(school_freqs, axis = 0)


    def graph_values(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(ncols = 5, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])
        for fish in self.fishes:
            ax0.scatter(fish.head_x, fish.head_y, c = np.linspace(0,1,num = len(fish.head_x)), s = 2)

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(range(len(self.school_center_x)), self.school_center_x)
        ax1.set_title("School X")

        ax2 = plt.subplot(gs[1,1])
        ax2.plot(range(len(self.school_center_y)), self.school_center_y)
        ax2.set_title("School Y")

        ax2 = plt.subplot(gs[2,1])
        ax2.plot(range(len(self.school_center_z)), self.school_center_z)
        ax2.set_title("School Z")

        ax3 = plt.subplot(gs[0,2])
        ax3.plot(range(len(self.group_speed)), self.group_speed)
        ax3.set_title("School Speed")

        ax5 = plt.subplot(gs[2,2])
        ax5.plot(range(len(self.polarization)), self.polarization)
        ax5.set_title("School Polarization")

        ax6 = plt.subplot(gs[0,3])
        ax6.plot(range(len(self.school_areas)), self.school_areas)
        ax6.set_title("School Area")

        ax7 = plt.subplot(gs[1,3])
        ax7.plot(range(len(self.nearest_neighbor_distance)), self.nearest_neighbor_distance)
        ax7.set_title("School NND")

        ax8 = plt.subplot(gs[2,3])
        ax8.plot(range(len(self.school_groups)), self.school_groups)
        ax8.set_title("School Groups")

        plt.show()

class trial:
    def __init__(self, file_name, data_folder, n_fish = 8):
        self.file = file_name

        self.year = self.file[0:4]
        self.month = self.file[5:7]
        self.day = self.file[8:10]
        self.trial = self.file[11:13]
        self.abalation = self.file[15:16]
        self.darkness = self.file[18:19]
        self.flow = self.file[21:22]

        date = self.file[0:10]

        self.fish_type = species

        self.n_fish = n_fish
        self.data = pd.read_csv(data_folder+file_name, index_col=0, header=header)
        self.scorer = self.data.keys()[0][0]

        self.fishes = [fish_data("individual"+str(i+1),self.data,self.scorer,int(self.flow)) for i in range(n_fish)]

        #This sets the indexes so I can avoid any issues with having Fish 1 always be compared first 
        # and so on and so forth
        
        #Now we're going to do all the fish both ways to make NND stuff easier
        self.fish_comp_indexes = [[i,j] for i in range(n_fish) for j in range(i+1,n_fish)]

        for pair in self.fish_comp_indexes:
            random.shuffle(pair)

        # self.fish_comp_indexes = [[i,j] for i in range(n_fish) for j in range(n_fish)]

        # #Remove matched pairs
        # matched_index = [0,9,18,27,36,45,56,63]

        # self.fish_comp_indexes = np.delete(self.fish_comp_indexes, matched_index, axis = 0)

        self.fish_comps = [[0 for j in range(self.n_fish)] for i in range(self.n_fish)]

        #Now we fill in based on the randomized pairs
        for pair in self.fish_comp_indexes:
            self.fish_comps[pair[0]][pair[1]] = fish_comp(self.fishes[pair[0]],self.fishes[pair[1]])

        self.school_comp = school_comps(self.fishes, n_fish = n_fish, flow = int(self.flow))

    def return_trial_vals(self):
        print(self.year,self.month,self.day,self.trial,self.abalation,self.darkness,self.flow)

    def return_tailbeat_lens(self):
        all_tailbeat_lens = []

        for fish in self.fishes:
            all_tailbeat_lens.extend(np.diff(fish.zero_crossings))

        return all_tailbeat_lens

    def return_fish_lens(self):
        all_fish_lens = []

        for fish in self.fishes:
            all_fish_lens.extend(get_fish_length(fish))

        return all_fish_lens

    def return_raw_points(self):
        firstfish = True

        for fish in self.fishes:

            data_len = len(fish.head_x)

            d = {'Year': np.repeat(self.year,data_len),
                 'Month': np.repeat(self.month,data_len),
                 'Day': np.repeat(self.day,data_len),
                 'Trial': np.repeat(self.trial,data_len), 
                 'Tailbeat': np.linspace(1,data_len,data_len) // tailbeat_len,
                 'Frame': np.linspace(1,data_len,data_len),
                 'Ablation': np.repeat(self.abalation,data_len), 
                 'Darkness': np.repeat(self.darkness,data_len), 
                 'Flow': np.repeat(self.flow,data_len), 
                 'Species': np.repeat(self.fish_type,data_len), 
                 #get into cm, not BL or meters
                 'Head_X': fish.head_x, #* fish_len*100,
                 'Head_Y': fish.head_y, #* fish_len*100,
                 'Head_Z': fish.head_z, #* fish_len*100,
                 'Midline_X': fish.head_x, #* fish_len*100,
                 'Midline_Y': fish.head_y, #* fish_len*100,
                 'Midline_Z': fish.head_z} #* fish_len*100} #,

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)


    def return_comp_vals(self):
        firstfish = True

        for pair in self.fish_comp_indexes:

            current_comp = self.fish_comps[pair[0]][pair[1]]

            chunked_x_diffs = mean_tailbeat_chunk(current_comp.x_diff,tailbeat_len)
            chunked_y_diffs = mean_tailbeat_chunk(current_comp.y_diff,tailbeat_len)
            chunked_z_diffs = mean_tailbeat_chunk(current_comp.z_diff,tailbeat_len)
            chunked_dists = get_dist_np_3D(0,0,0,chunked_x_diffs,chunked_y_diffs,chunked_z_diffs)
            chunked_angles = angular_mean_tailbeat_chunk(current_comp.angle,tailbeat_len)
            chunked_yaw_heading_diffs = angular_mean_tailbeat_chunk(current_comp.yaw_heading_diff,tailbeat_len)
            chunked_pitch_heading_diffs = angular_mean_tailbeat_chunk(current_comp.pitch_heading_diff,tailbeat_len)
            chunked_f1_speed = mean_tailbeat_chunk(current_comp.f1.speed,tailbeat_len)
            chunked_f2_speed = mean_tailbeat_chunk(current_comp.f2.speed,tailbeat_len)
            chunked_f1_accel = current_comp.f1.accel
            chunked_f2_accel = current_comp.f2.accel
            chunked_f1_X = mean_tailbeat_chunk(current_comp.f1.head_x,tailbeat_len)
            chunked_f1_Y = mean_tailbeat_chunk(current_comp.f1.head_y,tailbeat_len)
            chunked_f1_Z = mean_tailbeat_chunk(current_comp.f1.head_z,tailbeat_len)
            chunked_f2_X = mean_tailbeat_chunk(current_comp.f2.head_x,tailbeat_len)
            chunked_f2_Y = mean_tailbeat_chunk(current_comp.f2.head_y,tailbeat_len)
            chunked_f2_Z = mean_tailbeat_chunk(current_comp.f2.head_z,tailbeat_len)
            chunked_relative_x = mean_tailbeat_chunk(current_comp.relative_x,tailbeat_len)
            chunked_f1_yaw_heading = angular_mean_tailbeat_chunk(current_comp.f1.yaw_heading,tailbeat_len)
            chunked_f2_yaw_heading = angular_mean_tailbeat_chunk(current_comp.f2.yaw_heading,tailbeat_len)
            chunked_f1_yaw_heading_ps = current_comp.f1.yaw_heading_ps
            chunked_f2_yaw_heading_ps = current_comp.f2.yaw_heading_ps
            chunked_speed_diffs = mean_tailbeat_chunk(current_comp.speed_diff,tailbeat_len)
            #chunked_tailbeat_offsets = mean_tailbeat_chunk(current_comp.tailbeat_offset_reps,tailbeat_len)

            min_data_length = min([len(chunked_x_diffs),len(chunked_y_diffs),len(chunked_dists),
                                     len(chunked_angles),len(chunked_yaw_heading_diffs),len(chunked_pitch_heading_diffs),
                                     len(chunked_f1_yaw_heading_ps),len(chunked_f2_yaw_heading_ps),
                                     len(chunked_f1_accel),len(chunked_f2_accel),
                                     len(chunked_speed_diffs)])

            max_data_length = max([len(chunked_x_diffs),len(chunked_y_diffs),len(chunked_dists),
                                     len(chunked_angles),len(chunked_yaw_heading_diffs),len(chunked_pitch_heading_diffs),
                                     len(chunked_f1_yaw_heading_ps),len(chunked_f2_yaw_heading_ps),
                                     len(chunked_f1_accel),len(chunked_f2_accel),
                                     len(chunked_speed_diffs)])

            tailbeat_offset = (max_data_length - min_data_length) // 2

            tb_start = (max_data_length - min_data_length) // 2
            tb_stop = max_data_length - math.ceil((max_data_length - min_data_length) / 2)


            d = {'Year': np.repeat(self.year,min_data_length),
                 'Month': np.repeat(self.month,min_data_length),
                 'Day': np.repeat(self.day,min_data_length),
                 'Trial': np.repeat(self.trial,min_data_length), 
                 'Ablation': np.repeat(self.abalation,min_data_length), 
                 'Darkness': np.repeat(self.darkness,min_data_length), 
                 'Flow': np.repeat(self.flow,min_data_length), 
                 'Species': np.repeat(self.fish_type,min_data_length), 
                 'Fish': np.repeat(current_comp.name,min_data_length),
                 'Tailbeat_Num': range(1,min_data_length+1),
                 'X_Distance': array_shortener(chunked_x_diffs,min_data_length,max_data_length), 
                 'Y_Distance': array_shortener(chunked_y_diffs,min_data_length,max_data_length),
                 'Z_Distance': array_shortener(chunked_z_diffs,min_data_length,max_data_length),
                 'Distance': array_shortener(chunked_dists,min_data_length,max_data_length),
                 'Angle': array_shortener(chunked_angles,min_data_length,max_data_length),
                 'Yaw_Heading_Diff': array_shortener(chunked_yaw_heading_diffs,min_data_length,max_data_length),
                 'Pitch_Heading_Diff': array_shortener(chunked_pitch_heading_diffs,min_data_length,max_data_length),
                 'Fish1_Speed': array_shortener(chunked_f1_speed,min_data_length,max_data_length),
                 'Fish2_Speed': array_shortener(chunked_f2_speed,min_data_length,max_data_length),
                 'Fish1_Accel': array_shortener(chunked_f1_accel,min_data_length,max_data_length),
                 'Fish2_Accel': array_shortener(chunked_f2_accel,min_data_length,max_data_length),
                 'Fish1_X': array_shortener(chunked_f1_X,min_data_length,max_data_length),
                 'Fish1_Y': array_shortener(chunked_f1_Y,min_data_length,max_data_length),
                 'Fish1_Z': array_shortener(chunked_f1_Z,min_data_length,max_data_length),
                 'Fish2_X': array_shortener(chunked_f2_X,min_data_length,max_data_length),
                 'Fish2_Y': array_shortener(chunked_f2_Y,min_data_length,max_data_length),
                 'Fish2_Z': array_shortener(chunked_f2_Z,min_data_length,max_data_length),
                 'Relative_X': array_shortener(chunked_relative_x,min_data_length,max_data_length),
                 'Fish1_Yaw_Heading': array_shortener(chunked_f1_yaw_heading,min_data_length,max_data_length),
                 'Fish2_Yaw_Heading': array_shortener(chunked_f2_yaw_heading,min_data_length,max_data_length),
                 'Fish1_Yaw_Heading_PS': array_shortener(chunked_f1_yaw_heading_ps,min_data_length,max_data_length),
                 'Fish2_Yaw_Heading_PS': array_shortener(chunked_f2_yaw_heading_ps,min_data_length,max_data_length),
                 'Speed_Diff': array_shortener(chunked_speed_diffs,min_data_length,max_data_length),
                 'Fish1_Speed_SD': np.repeat(current_comp.f1.speed_sd,min_data_length),
                 'Fish1_Speed_CV': np.repeat(current_comp.f1.speed_COV,min_data_length)} #,
                 #'Sync': chunked_tailbeat_offsets[:short_data_length]}

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_school_vals(self):

        chunked_x_center = mean_tailbeat_chunk(self.school_comp.school_center_x,tailbeat_len)
        chunked_y_center = mean_tailbeat_chunk(self.school_comp.school_center_y,tailbeat_len)
        chunked_z_center = mean_tailbeat_chunk(self.school_comp.school_center_z,tailbeat_len)
        chunked_x_sd = mean_tailbeat_chunk(self.school_comp.school_x_sd,tailbeat_len)
        chunked_y_sd = mean_tailbeat_chunk(self.school_comp.school_y_sd,tailbeat_len)
        chunked_z_sd = mean_tailbeat_chunk(self.school_comp.school_z_sd,tailbeat_len)
        chunked_group_speed = mean_tailbeat_chunk(self.school_comp.group_speed,tailbeat_len)
        chunked_polarization = mean_tailbeat_chunk(self.school_comp.polarization,tailbeat_len)
        chunked_nnd = mean_tailbeat_chunk(self.school_comp.nearest_neighbor_distance,tailbeat_len)
        chunked_area = mean_tailbeat_chunk(self.school_comp.school_areas,tailbeat_len)
        chunked_groups = median_tailbeat_chunk(self.school_comp.school_groups,tailbeat_len)
        chunked_group_means = mean_tailbeat_chunk(self.school_comp.school_groups,tailbeat_len)
        chunked_height = mean_tailbeat_chunk(self.school_comp.school_height,tailbeat_len)
        chunked_tailbeat_freq = mean_tailbeat_chunk(self.school_comp.school_tailbeat_freq,tailbeat_len)

        min_data_length = min([len(chunked_x_center),len(chunked_y_center),len(chunked_x_sd),
                                 len(chunked_y_sd),len(chunked_group_speed),
                                 len(chunked_nnd),len(chunked_area),len(chunked_groups),len(chunked_height),
                                 len(chunked_tailbeat_freq)])

        max_data_length = max([len(chunked_x_center),len(chunked_y_center),len(chunked_x_sd),
                                 len(chunked_y_sd),len(chunked_group_speed),
                                 len(chunked_nnd),len(chunked_area),len(chunked_groups),len(chunked_height),
                                 len(chunked_tailbeat_freq)])

        tailbeat_offset = (max_data_length - min_data_length) // 2

        tb_start = (max_data_length - max_data_length) // 2
        tb_stop = max_data_length - math.ceil((max_data_length - min_data_length) / 2)

        d = {'Year': np.repeat(self.year,min_data_length),
             'Month': np.repeat(self.month,min_data_length),
             'Day': np.repeat(self.day,min_data_length),
             'Trial': np.repeat(self.trial,min_data_length),
             'Tailbeat_Num': range(1,min_data_length+1),
             'Ablation': np.repeat(self.abalation,min_data_length), 
             'Darkness': np.repeat(self.darkness,min_data_length), 
             'Species': np.repeat(self.fish_type,min_data_length), 
             'Flow': np.repeat(self.flow,min_data_length), 
             'X_Center': array_shortener(chunked_x_center,min_data_length,max_data_length),
             'Y_Center': array_shortener(chunked_y_center,min_data_length,max_data_length),
             'Y_Center': array_shortener(chunked_z_center,min_data_length,max_data_length),
             'X_SD': array_shortener(chunked_x_sd,min_data_length,max_data_length),
             'Y_SD': array_shortener(chunked_y_sd,min_data_length,max_data_length),
             'Z_SD': array_shortener(chunked_z_sd,min_data_length,max_data_length),
             'School_Polar': array_shortener(chunked_polarization,min_data_length,max_data_length),
             'School_Speed': array_shortener(chunked_group_speed,min_data_length,max_data_length),
             'NND': array_shortener(chunked_nnd,min_data_length,max_data_length),
             'Area': array_shortener(chunked_area,min_data_length,max_data_length),
             'Groups': array_shortener(chunked_groups,min_data_length,max_data_length),
             'Mean_Groups': array_shortener(chunked_group_means,min_data_length,max_data_length),
             'School_Height': array_shortener(chunked_height,min_data_length,max_data_length),
             'Tailbeat_Freq': array_shortener(chunked_tailbeat_freq,min_data_length,max_data_length)}

        out_data = pd.DataFrame(data=d)

        return(out_data)


data_folder = "3D_Finished_Fish_Data_4P_gaps/"+species+"/"
trials = []

single_file = "" #"2020_07_28_11" # "2020_07_28_03_LN_DN_F2" #"2021_10_06_36_LY_DN_F2_3D_DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv"

for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv") and single_file in file_name:
        print(file_name)

        trials.append(trial(file_name,data_folder))

first_trial = True

#pair = trials[0].fish_comp_indexes[3]
#trials[0].fish_comps[pair[0]][pair[1]].graph_values()

print("Creating CSVs...")

for trial in trials:
    if first_trial:
        fish_comp_dataframe = trial.return_comp_vals()
        fish_school_dataframe = trial.return_school_vals()
        fish_raw_dataframe = trial.return_raw_points()
        first_trial = False
    else:
        fish_comp_dataframe = fish_comp_dataframe.append(trial.return_comp_vals())
        fish_school_dataframe = fish_school_dataframe.append(trial.return_school_vals())
        fish_raw_dataframe = fish_raw_dataframe.append(trial.return_raw_points())

fish_comp_dataframe.to_csv("Fish_Comp_Values_3D_"+species+".csv")
fish_school_dataframe.to_csv("Fish_School_Values_3D_"+species+".csv")
fish_raw_dataframe.to_csv("Fish_Raw_Points_3D_"+species+".csv")


