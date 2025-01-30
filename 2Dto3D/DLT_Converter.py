
import os
import numpy as np
import pandas as pd
import math
#import plotly.express as px
from scipy import linalg
import sys

#Header list for reading the raw location CSVs
header = list(range(4))

#Get all te files
v1_filepath = "V1 CSVs/"
v2_filepath = "V2 CSVs/"
dlt_coef_filepath = "DLT Coefs/"

v1_files = os.listdir(v1_filepath)
v2_files = os.listdir(v2_filepath)
dlt_coef_files = os.listdir(dlt_coef_filepath)

num_fish = 8
body_parts = ["head","midline2"]

def DLTdvRecon(Ls, uvs):

    #uvs are in format [[v1_xs,v1_ys],[v2_xs,v2_ys]]

    Ls = np.array(Ls)
    uvs = np.array(uvs)

    #http://kwon3d.com/theory/dlt/dlt.html#3d

    #https://github.com/tlhedrick/dltdv/blob/master/DLTdv8a_internal/dlt_reconstruct_v2.m

    m1 = np.zeros((4,3))
    m2 = np.zeros((4,1))

    m1[0:3:2,0] = uvs[:,0] * Ls[:,8] - Ls[:,0]
    m1[0:3:2,1] = uvs[:,0] * Ls[:,9] - Ls[:,1]
    m1[0:3:2,2] = uvs[:,0] * Ls[:,10] - Ls[:,2]

    m1[1:4:2,0] = uvs[:,1] * Ls[:,8] - Ls[:,4]
    m1[1:4:2,1] = uvs[:,1] * Ls[:,9] - Ls[:,5]
    m1[1:4:2,2] = uvs[:,1] * Ls[:,10] - Ls[:,6]

    m2[0:3:2,0] = Ls[:,3] - uvs[:,0]
    m2[1:4:2,0] = Ls[:,7] - uvs[:,1]

    xyz = linalg.lstsq(m1, m2, lapack_driver = "gelsy")[0]

    return xyz


single_file =  "" #"2021_10_08_26_LN_DY_F0_V2DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered"

# We have more v1 files than v2, so we do this for every v2 file
for v2f in v2_files:
    if v2f.endswith(".csv") and single_file in v2f:

        #Get a long ID for the matching V1, short ID for the DLT
        file_id = v2f[0:22]
        short_id = v2f[0:10]

        print(short_id)

        #Get the v1 file that matches, and the dlt coefs that go with them both
        v1f = [f for f in v1_files if file_id in f][0]

        dlt_coefs_file = [f for f in dlt_coef_files if short_id in f][0]

        #Add the filepath on here as well
        v1f = v1_filepath + v1f
        v2f = v2_filepath + v2f
        dlt_coefs_file = dlt_coef_filepath + dlt_coefs_file

        print(v1f,v2f,dlt_coefs_file)

        #Read in the raw data
        v1_raw_data = pd.read_csv(v1f, index_col=0, header=header)
        v2_raw_data = pd.read_csv(v2f, index_col=0, header=header)
        dlt_coefs_data = pd.read_csv(dlt_coefs_file)

        dlt_coefs = np.zeros((2,11))
        dlt_coefs[0] = np.array(dlt_coefs_data["C1"])
        dlt_coefs[1] = np.array(dlt_coefs_data["C2"])

        v1_scorer = v1_raw_data.keys()[0][0]
        v2_scorer = v2_raw_data.keys()[0][0]

        fish_data_out_dict = {}

        for fn in range(1,num_fish+1):
            name = "individual"+str(fn)

            for bp in body_parts:

                #Get the data for the individual and body part
                v1_x = v1_raw_data[v1_scorer][name][bp]["x"].to_numpy() 
                v1_y = v1_raw_data[v1_scorer][name][bp]["y"].to_numpy() 

                v2_x = v2_raw_data[v2_scorer][name][bp]["x"].to_numpy() 
                v2_y = v2_raw_data[v2_scorer][name][bp]["y"].to_numpy() 

                #Create the dict using tuples as keys to work better with pandas
                fish_data_out_dict[(v1_scorer,name,bp,"x")] = np.zeros(min(len(v1_x),len(v2_x)))
                fish_data_out_dict[(v1_scorer,name,bp,"y")] = np.zeros(min(len(v1_x),len(v2_x)))
                fish_data_out_dict[(v1_scorer,name,bp,"z")] = np.zeros(min(len(v1_x),len(v2_x)))

                for i in range(min(len(v1_x),len(v2_x))):

                    points_array = [[v1_x[i],v1_y[i]],[v2_x[i],v2_y[i]]]

                    if not np.isnan(np.sum(points_array)):
                        #Convert to 3D
                        #Not on pixel scale, fix later
                        points_3D = DLTdvRecon(dlt_coefs,points_array)

                        #Put the data in the right places
                        fish_data_out_dict[(v1_scorer,name,bp,"x")][i] = points_3D[0][0]
                        fish_data_out_dict[(v1_scorer,name,bp,"y")][i] = points_3D[1][0]
                        fish_data_out_dict[(v1_scorer,name,bp,"z")][i] = points_3D[2][0]

                    else:
                        #This makes them blanks
                        fish_data_out_dict[(v1_scorer,name,bp,"x")][i] = np.nan
                        fish_data_out_dict[(v1_scorer,name,bp,"y")][i] = np.nan
                        fish_data_out_dict[(v1_scorer,name,bp,"z")][i] = np.nan      

        fish_out_df = pd.DataFrame.from_dict(fish_data_out_dict)
        #Replace to make new file name
        new_file_name = v2f.replace(v2_filepath,"").replace("V2","3D_").replace("TN","LN")
        fish_out_df.to_csv("/Users/Ben/Desktop/Ablation-Species-Code/2Dto3D/Final 3D/"+new_file_name)







