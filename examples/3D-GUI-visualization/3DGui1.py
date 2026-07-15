import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

#source_path=r"C:\Users\asus\Desktop\zeynep calisma\neuroscience drosophila\sequential-inverse-kinematics\examples"
source_path=r".."
if source_path not in sys.path:
    sys.path.append(source_path)

from seqikpy.utils import load_file, save_file, calculate_body_size, dict_to_nparray_pose

from seqikpy.alignment import AlignPose, convert_from_df3dpp_to_dict, convert_from_anipose_to_dict
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq
from seqikpy.visualization import plot_3d_points, animate_3d_points,generate_color_map
from seqikpy.body_config import neuromechfly_body_config
from seqikpy.head_inverse_kinematics import HeadInverseKinematics

print("seqIKPy succesfully imported.")

# Set up the constant variables
leg_joint_angle_names = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]
legs_to_align_locomotion = ["RF", "RM", "RH", "LF", "LM", "LH"]
legs_to_align_grooming = ["RF","LF"]

keypoints=["ThC","CTr","FTi","TiTa","Claw"]

#loading processed sample data
#import_folder=Path(r"C:\Users\asus\Desktop\zeynep calisma\neuroscience drosophila\sequential-inverse-kinematics\data\inverse-kinematics-processed-data-for-comparison\Visualization-data")
import_folder=Path(r"..\..\data\inverse-kinematics-processed-data-for-comparison\Visualization-data")

#inverse kinematic outputs    
locomotion_leg_joint_angles=load_file(import_folder/"locomotion_leg_joint_angles_processed_300f.pkl")
locomotion_forward_kinematics=load_file(import_folder/"locomotion_forward_kinematics_processed_300f.pkl")

grooming_leg_joint_angles=load_file(import_folder/"grooming_leg_joint_angles_processed_300f.pkl")
grooming_forward_kinematics=load_file(import_folder/"grooming_forward_kinematics_processed_300f.pkl")

#raw aligned pose
aligned_pos_locomotion=load_file(import_folder/"locomotion_aligned_pos_300f.pkl")
aligned_pos_grooming=load_file(import_folder/"grooming_aligned_pos_300f.pkl")

print("Sample files succesfully imported")

def df_convert(points_data_dict,line_type):
    rows=[]

    for leg_name,array in points_data_dict.items():
        if not array.shape[0]==1:
            num_frames=array.shape[0]
            num_keypoints=array.shape[1]

            for t in range(num_frames):
                for kp_idx in range(num_keypoints):
                    if num_keypoints==9:
                        if kp_idx<4:
                            kp_name=keypoints[0]
                        elif kp_idx<6:
                            kp_name=keypoints[1]
                        else:
                            kp_name=keypoints[kp_idx-4]

                    elif num_keypoints==5:
                        kp_name=keypoints[kp_idx]
                    else:
                        kp_name=f"Keypoint_{kp_idx}"

                    x = array[t, kp_idx, 0]
                    y = array[t, kp_idx, 1]
                    z = array[t, kp_idx, 2]

                    rows.append({
                        "frame":t,
                        "leg_name":leg_name,
                        "keypoint_name":kp_name,
                        "x":x,
                        "y":y,
                        "z":z,
                        "line_type":line_type,
                        "leg+line":f"{leg_name}_{line_type}"
                    })

        elif array.shape[0]==1:
            num_frames=max([array.shape[0] for leg_name,array in points_data_dict.items()])
            num_keypoints=array.shape[1]
            for t in range(num_frames):
                for kp_idx in range(num_keypoints):
                    if num_keypoints==9:
                        if kp_idx<4:
                            kp_name=keypoints[0]
                        elif kp_idx<6:
                            kp_name=keypoints[1]
                        else:
                            kp_name=keypoints[kp_idx-4]

                    elif num_keypoints==5:
                        kp_name=keypoints[kp_idx]
                    else:
                        kp_name=f"Keypoint_{kp_idx}"

                    x = array[0, kp_idx, 0]
                    y = array[0, kp_idx, 1]
                    z = array[0, kp_idx, 2]

                    rows.append({
                        "frame":t,
                        "leg_name":leg_name,
                        "keypoint_name":kp_name,
                        "x":x,
                        "y":y,
                        "z":z,
                        "line_type":line_type,
                        "leg+line":f"{leg_name}_{line_type}"
                    })

            

    return pd.DataFrame(rows), num_frames



st.set_page_config(page_title="3D Visualization of Drosophila Melanogaster", layout="wide")
st.title("3D Visualization of Drosophila Melanogaster")
left_space, right_space = st.columns([1, 1])

#3D grid 
with left_space:

    selection = st.selectbox(
        "Select the pose to visualize",
        options=[
            "Locomotion - Forward Kinematics Sample",
            "Grooming - Forward Kinematics Sample",
            "Visualize my uploaded pose data"
        ])
    
    

    fig=go.Figure()
    
    def visualize(poseData3D,poseDataDashed=None):
        df_converted, num_frames=df_convert(poseData3D,"solid")
        if poseDataDashed is not None:
            df_converted_fk, num_frames_fk=df_convert(poseDataDashed,"dash")
            df_converted= pd.concat([df_converted, df_converted_fk], ignore_index=True)

        #color_map_right=generate_color_map("Reds", len(poseData3D)+1)
        #color_map_left=generate_color_map("Blues", len(poseData3D)+1)
        
        def get_color(leg_name):
            if "RF" in leg_name: return "rgb(255, 120, 120)" 
            elif "RM" in leg_name: return "rgb(250, 40, 40)"
            elif "RH" in leg_name: return "rgb(120, 0, 0)"

            elif "LF" in leg_name: return "rgb(130, 210, 255)"
            elif "LM" in leg_name: return "rgb(0, 120, 255)"
            elif "LH" in leg_name: return "rgb(0, 45, 250)" 
            return "rgb(225,225,225)"

        init_df = df_converted[df_converted["frame"] == 0]
        for leg_name, data in init_df.groupby("leg+line"):

            hover_texts = [f"{row["leg_name"]} - {row['keypoint_name']}" for idx, row in data.iterrows()]

            fig.add_trace(go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="lines+markers",
                line=dict(color=get_color(leg_name), width=4,dash=data["line_type"].iloc[0]),
                marker=dict(size=3),
                name=data["leg_name"].iloc[0],
                showlegend=("solid" in leg_name),
                hovertext=hover_texts,
                hoverinfo="text+x+y+z"


            ))

        frames = []
        for t in range(num_frames):
            frame_df = df_converted[df_converted["frame"] == t]
            frame_data = []
            
            for leg_name, data in frame_df.groupby("leg+line"):
                frame_data.append(go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],

                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))

        fig.frames = frames

        # Interface
        updatemenus = [dict(
            type="buttons",
            buttons=[
                dict(label="▶️ Play", method="animate",
                    args=[None, dict(frame=dict(duration=10, redraw=True), fromcurrent=True)]),
                dict(label="⏸️ Pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left", pad={"r": 10, "t": 87}, x=0.1, xanchor="right", y=0, yanchor="top"
        )]

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Frame (t): ", "font": {"size": 14}},
            pad={"b": 10, "t": 50}, len=0.9, x=0.1, y=0,
            steps=[dict(label=str(t), method="animate",
                        args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for t in range(0,num_frames)]
        )]


        #Layout
        fig.update_layout(
            width=800,
            height=700,
            scene=dict(
                xaxis=dict(range=[-4, 4], title="X"),
                yaxis=dict(range=[-4, 4], title="Y"),
                zaxis=dict(range=[-3, 3], title="Z"),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.8),
                camera= dict(
                    up=dict(x=0, y=0, z=1),       
                    center=dict(x=0, y=0, z=0),   
                    eye=dict(x=0.75, y=0.75, z=0.7))
            
            ),
            updatemenus=updatemenus,
            sliders=sliders,
            legend=dict(x=1.05, y=0.8)
        )


    if selection == "Locomotion - Forward Kinematics Sample":
        poseData3D= locomotion_forward_kinematics
        AlignedPose3D=aligned_pos_locomotion

        st.info("Visualizing Locomotion data sample  from SeqIKPy forward kinematics\n\n____ Solid lines: Raw pose\n\n------- Dashed lines: SeqIKPy Forward Kinematics")
        visualize(AlignedPose3D,poseData3D)

    elif selection == "Grooming - Forward Kinematics Sample":
        poseData3D= grooming_forward_kinematics
        AlignedPose3D=aligned_pos_grooming
        st.info("Visualizing Grooming data sample from SeqIKPy forward kinematics\n\n____ Solid lines: Raw pose\n\n------- Dashed lines: SeqIKPy Forward Kinematics")
        visualize(AlignedPose3D,poseData3D)


    elif selection== "Visualize my uploaded pose data":
        pose_type_selection = st.selectbox(
        "Which type of pose would you like to visualize?",
        options=[
            "Aligned Pose Data",
            "Forward Kinematics",
            "Both"
        ])

        if (pose_type_selection=="Aligned Pose Data") or (pose_type_selection=="Forward Kinematics"):
            uploaded_file= st.file_uploader(
                label=f"Load your {pose_type_selection} data to visualize 3d pose", 
                type=["pkl","h5","hdf","npy"])
            
            if uploaded_file is not None:
                try:
                    poseData3D = pickle.load(uploaded_file)
                    st.info(f"Visualizing {uploaded_file.name}")
                    visualize(poseData3D)
                except Exception as e:
                    st.error(f"❌ Cannot visualize data: {e}")



        elif pose_type_selection=="Both":
            uploaded_file_AL= st.file_uploader(
                label=f"Load your aligned pose data here (Solid lines)", 
                type=["pkl","h5","hdf","npy"])
            
            
            uploaded_file_FK= st.file_uploader(
                label=f"Load your forward kinematics data here (Dashed lines)", 
                type=["pkl","h5","hdf","npy"])
            
            if (uploaded_file_AL is not None) and (uploaded_file_FK is not None):
                try:
                    poseData3D = pickle.load(uploaded_file_AL)
                    poseData3D_fk = pickle.load(uploaded_file_FK)
                    st.info(f"Visualizing your data")
                    visualize(poseData3D,poseData3D_fk)
                except Exception as e:
                    st.error(f"❌ Cannot visualize data: {e}")

    st.plotly_chart(fig, use_container_width=True)


