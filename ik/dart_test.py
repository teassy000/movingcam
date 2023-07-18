import numpy as np
#import pydart2 as pydart
import dartpy as dart



import sys
sys.path.append('C:/work/pose_estimation/movingcam')


from PyCommon.modules.GUI import DartViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Math import mmMath as mm
from fltk import Fl
import math
from scipy.optimize import minimize, Bounds
import joblib
import subprocess

from SkateUtils.DartMotionEdit import skelqs2bvh, DartSkelMotion

def get_spin_joint_names():
    # vibe 3d joint indices
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]


def get_staf_joint_names():
    # vibe joints2d output
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]

name_idx_map = {
    'j_root': 39,
    # 'j_root_2':8

    'j_thigh_left': 28,
    'j_shin_left': 13, 
    # 'j_shin_left_2': 29,
    'j_heel_left': 14,
    # 'j_heel_left_2': 30,
    'j_bigtoe_left': 19,
    'j_smalltoe_left': 20,
    'j_trueheel_left': 21,
    
    'j_thigh_right': 27,
    'j_shin_right': 10,
    # 'j_shin_right_2': 26,
    'j_heel_right': 11,
    # 'j_heel_right_2': 25,
    'j_bigtoe_right': 22,
    'j_smalltoe_right': 23,
    'j_trueheel_right': 24,
    
    # 'j_spine': -1,
    
    'j_neck': 1,
    # 'j_head': -1,
    
    'j_bicep_left': 5,
    # 'j_bicep_left_2': 34,
    'j_forearm_left': 6,
    # 'j_forearm_left_2': 35,
    'j_hand_left': 7,
    # 'j_hand_left_2': 36,

    'j_bicep_right': 2,
    # 'j_bicep_right_2': 33,
    'j_forearm_right': 3,
    # 'j_forearm_right_2': 32,
    'j_hand_right': 4,
    # 'j_hand_right_2': 31,
    }

default_height = 10.0  # m
default_width = 0.2  # m
default_depth = 0.2  # m

default_torque = 15.0  # N-m
default_force = 15.0  # N
default_countdown = 200  # Number of timesteps for applying force

default_rest_position = 0.0
delta_rest_position = 10.0 * math.pi / 180.0

default_stiffness = 0.0
delta_stiffness = 10

default_damping = 5.0
delta_damping = 1.0


def set_geometry(body):
    # Create a BoxShape to be used for both visualization and collision checking
    box = dart.dynamics.BoxShape([default_width, default_depth, default_height])

    # Create a shape node for visualization and collision checking
    shape_node = body.createShapeNode(box)
    visual = shape_node.createVisualAspect()
    shape_node.createCollisionAspect()
    shape_node.createDynamicsAspect()
    visual.setColor([0, 0, 1])

    # Set the location of the shape node
    box_tf = dart.math.Isometry3()
    center = [0, 0, default_height / 2.0]
    box_tf.set_translation(center)
    shape_node.setRelativeTransform(box_tf)

    # Move the center of mass to the center of the object
    body.setLocalCOM(center)


def make_root_body(chain, name):
    joint_prop = dart.dynamics.BallJointProperties()
    joint_prop.mName = name + "_joint"
    joint_prop.mRestPositions = np.ones(3) * default_rest_position
    joint_prop.mSpringStiffnesses = np.ones(3) * default_stiffness
    joint_prop.mDampingCoefficients = np.ones(3) * default_damping

    body_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name)
    body_prop = dart.dynamics.BodyNodeProperties(body_aspect_prop)

    [joint, body] = chain.createBallJointAndBodyNodePair(None, joint_prop, body_prop)

    # Make a shape for the Joint
    r = default_width
    ball = dart.dynamics.EllipsoidShape(math.sqrt(2) * np.ones(3) * r)
    shape_node = body.createShapeNode(ball)
    visual = shape_node.createVisualAspect()
    visual.setColor([0, 0, 1, 0.8])

    # Set the geometry of the BodyNode
    set_geometry(body)

    return body


def add_body(chain, parent, name):
    # Set up the properties for the Joint
    joint_prop = dart.dynamics.RevoluteJointProperties()
    joint_prop.mName = name + "_joint"
    joint_prop.mAxis = [0, 1, 0]
    joint_prop.mT_ParentBodyToJoint.set_translation([0, 0, default_height])

    # Set up the properties for the BodyNode
    body_aspect_prop = dart.dynamics.BodyNodeAspectProperties(name)
    body_prop = dart.dynamics.BodyNodeProperties(body_aspect_prop)

    # Create a new BodyNode, attached to its parent by a RevoluteJoint
    [joint, body] = chain.createRevoluteJointAndBodyNodePair(
        parent, joint_prop, body_prop
    )

    joint.setRestPosition(0, default_rest_position)
    joint.setSpringStiffness(0, default_stiffness)
    joint.setDampingCoefficient(0, default_damping)

    # Make a shape for the Joint
    r = default_width / 2.0
    h = default_depth
    cylinder = dart.dynamics.CylinderShape(r, h)

    # Line up the cylinder with the Joint axis
    tf = dart.math.Isometry3()
    angles = [math.pi / 2, 0, 0]

    rot = dart.math.eulerXYZToMatrix(angles)
    tf.set_rotation(rot)

    shape_node = body.createShapeNode(cylinder)
    visual = shape_node.createVisualAspect()
    visual.setColor([0, 0, 1, 0.8])
    shape_node.setRelativeTransform(tf)

    # Set the geometry of the Body
    set_geometry(body)

    return body



class MyWorldNode(dart.gui.RealTimeWorldNode):
    def __init__(self, world):
        super(MyWorldNode, self).__init__(world)
        pass

    def customPreStep(self):
        pass

if __name__ == "__main__":
    # video_name = input("video name?")
    
    video_name = 'ollie'
    base_path = 'C:/work/pose_estimation/movingcam'

    frame_rate = subprocess.check_output(f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 {base_path}/data/openpose/{video_name}/{video_name}.mp4", shell=True)
    frame_rate = list(map(float, frame_rate.decode().split('/')))
    fps = int(frame_rate[0] / frame_rate[1] + 0.1)
    print("fps: ", fps)

    file_path = base_path + '/data/vibe/' + video_name +'/vibe_output.pkl'
    data = joblib.load(file_path)
    print("data length:",len(data))
    print("keys: ", data.keys())

    frame_ids = list(map(int, data[1]['frame_ids']))

    frame_num = int(np.max(frame_ids))+1
    missing_ranges = []
    for _i in range(len(frame_ids)-1):
        if frame_ids[_i+1] - frame_ids[_i] > 1:
            missing_ranges.append((frame_ids[_i], frame_ids[_i+1]))


    joint_3d_infos = data[1]['joints3d']
    
    joint_num = len(joint_3d_infos[0])

    print("frame_num:", frame_num)
    print("joint_num:", joint_num)
    print(joint_3d_infos.shape)

    world = dart.io.SkelParser.readWorld('data/skel/human_mass_limited_dof_v2.skel')
    world.setGravity([0, -9.81, 0])

    skel = world.getSkeleton(1)

    #world = dart.simulation.World()
    #world.addSkeleton(skel)


    # Create world node and add it to viewer
    viewer = dart.gui.Viewer()
    
    shadow = dart.gui.WorldNode.createDefaultShadowTechnique(viewer)


    node = MyWorldNode(world)
    viewer.addWorldNode(node)

    grid = dart.gui.GridVisual()
    grid.setPlaneType(dart.gui.GridVisual.PlaneType.ZX)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)


    viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setCameraHomePosition([8.0, 8.0, 4.0], [0, 0, 0], [0, 1, 0])
    
    viewer.simulate(True)
    frameIdx = 0

    missing_frames = []

    motion = DartSkelMotion()
    motion.fps = fps

    while(frameIdx < frame_num):
        viewer.frame(frameIdx)
        frameIdx += 1
        print(frame_ids)
        if frameIdx in frame_ids:
            data_frame = frame_ids.index(frameIdx)
        else:
            missing_frames.append(frameIdx)
            #motion.append(skel.getPosition(), skel.getVelocity())
            continue
    

