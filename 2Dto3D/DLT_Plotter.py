import os
import numpy as np
import pandas as pd
import math
import plotly.express as px
from scipy import linalg
import sys

#Header list for reading the raw location CSVs
header = list(range(4))

fish_len = 0.083197

num_fish = 8
body_parts = ["head","midline2"]

file_folder = "Final 3D/"

three_d_files = os.listdir(file_folder)

for file_name in three_d_files:
    if file_name.endswith(".csv"): #and "03_25_10" in file_name:

        print(file_name)

        file_path = file_folder+file_name

        fish_raw_data = pd.read_csv(file_path, index_col=0, header=header)

        #They asked me why by god would I ever do this in pandas and not dplyr like I ACTUALLY KNOW HOW TO DO
        # but plotly is better and I have code for that, and by god I will learn how to use pandas

        #So first we melt it to make all the column names into rows
        #This is great! Other than we still want x, y, and z to be their own columns and not just labeled
        df = pd.melt(fish_raw_data)

        #Let's rename things to what they actually are
        df = df.rename(columns={"variable_0": "Scorer", "variable_1": "Fish", "variable_2": "BodyPart", "variable_3": "Point"})

        #So now I add the Frame to this because I will need to group by that otherwise there is overlap between labels
        df['Frame'] = df.groupby(['Scorer','Fish','BodyPart','Point']).cumcount()+1

        #So now we pivot from the Point column, into seperate ones for x, y, and z
        df = df.pivot_table(index = ['Scorer','Fish','BodyPart','Frame'], columns='Point', values='value')

        #Then we have to reset the index so that we can use 'Scorer','Fish','BodyPart','Frame' as data columns
        df = df.reset_index()

        #THANKS STACK EXCHANGE WOOOOOOO

        #Now we graph it, nice and easy

        # print(df)

        # df.to_csv()

        # sys.exit()

        #Though the axies keep shifting weirdly so not my fave
        fig = px.scatter_3d(df,x="x", y="y", z="z", color="Fish", animation_frame="Frame", hover_data = ["BodyPart"],
                               range_x=[df["x"].min()-0.05,df["x"].max()+0.05],
                               range_y=[df["y"].min()-0.05,df["y"].max()+0.05],
                               range_z=[df["z"].min()-0.05,df["z"].max()+0.05],
                               color_continuous_scale = "rainbow")

        fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}

        #fig.show()

        file_id = file_name[0:22]

        #fig.write_html("Saved 3D Plots/{name}_animated.html".format(name = file_id), auto_play=False)

        head_df = df[df['BodyPart'] == "head"] 

        head_df["x"] = head_df["x"]
        head_df["y"] = head_df["y"]
        head_df["z"] = head_df["z"]

        fig = px.line_3d(head_df,x="x", y="y", z="z", color="Fish", hover_data = ["BodyPart"])

        fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}

        #fig.show()

        fig.write_html("Saved 3D Plots/{name}_trace.html".format(name = file_id), auto_play=False)
