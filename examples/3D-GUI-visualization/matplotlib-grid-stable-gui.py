import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#source_path=r"C:\Users\asus\Desktop\zeynep calisma\neuroscience drosophila\sequential-inverse-kinematics\examples"
source_path=r".."
if source_path not in sys.path:
    sys.path.append(source_path)

from seqikpy.utils import load_file, save_file, calculate_body_size, dict_to_nparray_pose

from seqikpy.alignment import AlignPose, convert_from_df3dpp_to_dict, convert_from_anipose_to_dict
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq
from seqikpy.visualization import plot_3d_points, animate_3d_points
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

#loading processed sample data
#import_folder=Path(r"C:\Users\asus\Desktop\zeynep calisma\neuroscience drosophila\sequential-inverse-kinematics\data\inverse-kinematics-processed-data-for-comparison\Visualization-data")
import_folder=Path(r"..\..\data\inverse-kinematics-processed-data-for-comparison\Visualization-data")

#inverse kinematic outputs    
locomotion_leg_joint_angles=load_file(import_folder/"locomotion_leg_joint_angles_processed_300f.pkl")
locomotion_forward_kinematics=load_file(import_folder/"locomotion_forward_kinematics_processed_300f.pkl")

grooming_leg_joint_angles=load_file(import_folder/"grooming_leg_joint_angles_processed_300f.pkl")
grooming_forward_kinematics=load_file(import_folder/"grooming_forward_kinematics_processed_300f.pkl")

#original aligned posture
aligned_pos_locomotion=load_file(import_folder/"locomotion_aligned_pos_300f.pkl")
grooming_pos_locomotion=load_file(import_folder/"grooming_aligned_pos_300f.pkl")
print("Sample files succesfully imported")

st.set_page_config(page_title="3D Visualization of Drosophila Melanogaster", layout="wide")
st.title("3D Visualization of Drosophila Melanogaster")
left_space, right_space = st.columns([1, 1])

#3D grid 
with left_space:
    plt.style.use('default')

    t = st.slider(
        label="Frame(t):", 
        min_value=0, 
        max_value=299, 
        value=0
    )
    azim=30
    elev=10

    fig = plt.figure(figsize=(9,7),dpi=100)
    ax3d = fig.add_subplot(projection='3d')
    ax3d.view_init(azim=azim, elev=elev)

    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')

    ax3d.set_xlim([-2.5, 2.5])
    ax3d.set_ylim([-2, 2])
    ax3d.set_zlim([-2,1])
    
    plot_3d_points(
        ax3d,
        aligned_pos_locomotion,
        t=t,
        line_style='solid'
        )
    ax3d.legend(bbox_to_anchor=(1.3,0.6))

    plot_3d_points(
        ax3d,
        locomotion_forward_kinematics, 
        t=t,
        line_style='--'
    )
    
    ax3d.set_title('Solid - raw 3D, Dashed - FK', y=0.95)
    st.pyplot(fig)

