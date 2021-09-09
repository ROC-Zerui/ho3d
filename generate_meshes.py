"""
Generate mesh from ho3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
import os
import open3d

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        from pip._internal.main import main as pipmain
        pipmain(['install', package])

try:
    import matplotlib.pyplot as plt
except:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import chumpy as ch
except:
    install('chumpy')
    import chumpy as ch

try:
    import pickle
except:
    install('pickle')
    import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D


MANO_MODEL_PATH = './mano/models/MANO_RIGHT.pkl'
MANO_WATERTIGHT_OBJ_PATH = './mano/models/MANO_WATERTIGHT.obj'

if not os.path.exists(MANO_MODEL_PATH):
    raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
    from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model

if not os.path.exists(MANO_WATERTIGHT_OBJ_PATH): 
    raise Exception('watertight MANO object  missing! Please create watertight MANO obj')

def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m


def mesh2open3dMesh2npy(mList, colorList, mesh_category = None):
    import open3d
    o3dMeshList = []
    o3dFacesList = []
    for i, m in enumerate(mList):
        mesh = open3d.geometry.TriangleMesh()
        numVert = 0
        if hasattr(m, 'r'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.r))
            numVert = m.r.shape[0]
        elif hasattr(m, 'v'):
            mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.v))
            numVert = m.v.shape[0]
        else:
            raise Exception('Unknown Mesh format')
  
        if mesh_category == 'hand':
            mano_watertight_mesh = open3d.io.read_triangle_mesh(MANO_WATERTIGHT_OBJ_PATH)
            mesh.triangles = open3d.utility.Vector3iVector(np.copy(np.asarray(mano_watertight_mesh.triangles)))
        elif mesh_category == 'obj':
            mesh.triangles = open3d.utility.Vector3iVector(np.copy(m.f))
        else:
            raise Exception("missing mesh_category argument, should be 'hand' or 'obj' ")


        if colorList[i] == 'r':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
        elif colorList[i] == 'g':
            mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
        else:
            raise Exception('Unknown mesh color')

        o3dMeshList.append(mesh)
        o3dFacesList.append(np.asarray(mesh.vertices)[np.asarray(mesh.triangles)])
    
    # uncomment below line to visualize
    # open3d.visualization.draw_geometries(o3dMeshList)
    # return triangles with co-ordnates(#triangles x 3 (vertices) x 3 (xyz coordinate of each vertex))
    return o3dFacesList, o3dMeshList


def normalize_center_joint_mesh(o3d_mesh_hand, o3d_mesh_obj):
    # hand mesh vertices mean
    o3d_mesh_hand_mean_v = np.asarray(np.mean(o3d_mesh_hand.vertices, axis=0)).reshape((1, -1))
    
    # hand mesh vertices mean
    o3d_mesh_obj_mean_v = np.asarray(np.mean(o3d_mesh_obj.vertices, axis=0)).reshape((1, -1))
    # raw_input("enter for hand and obj  v mean")
    print("o3d_mesh_hand_mean_v", o3d_mesh_hand_mean_v)
    print("o3d_mesh_obj_mean_v", o3d_mesh_obj_mean_v)
    
    centered_hand_mesh = open3d.geometry.TriangleMesh()
    centered_object_mesh = open3d.geometry.TriangleMesh()
    
    # Non-weighted average
    vertices_mean = (o3d_mesh_hand_mean_v + o3d_mesh_obj_mean_v) / 2.0
    # raw_input("enter for non-weighted vertices_mean")
    print("Non weighted_vertices_mean", vertices_mean)

    centered_hand_mesh.vertices = open3d.utility.Vector3dVector(np.copy(np.subtract(np.asarray(o3d_mesh_hand.vertices), vertices_mean)))  
    centered_hand_mesh.triangles = open3d.utility.Vector3iVector(np.copy(np.asarray(o3d_mesh_hand.triangles)))
    numVert = np.asarray(o3d_mesh_hand.vertices).shape[0]
    centered_hand_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))

    centered_object_mesh.vertices = open3d.utility.Vector3dVector(np.copy(np.subtract(np.asarray(o3d_mesh_obj.vertices), vertices_mean)))  
    centered_object_mesh.triangles = open3d.utility.Vector3iVector(np.copy(np.asarray(o3d_mesh_obj.triangles)))
    numVert = np.asarray(o3d_mesh_obj.vertices).shape[0]
    centered_object_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))

    return centered_hand_mesh, centered_object_mesh


if __name__ == '__main__':
    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-meta_data_path", type=str, default="/home2/zerui/dataset/HO3D_V3/debug",
    help="Path to HO3D dataset train or val or test directory", required=False)
    ap.add_argument("-ycbModels_path", type=str, default="/home2/zerui/dataset/HO3D_V3",
    help="Path to ycb models directory", required=False)

    args = vars(ap.parse_args())

    baseDir = args['meta_data_path'] 
    YCBModelsDir = args['ycbModels_path']

    for seq in os.listdir(baseDir):
        for filename in os.listdir(os.path.join(baseDir, seq, 'meta')):
            print("filepath:", os.path.join(baseDir, seq, 'meta',  filename))
            anno = load_pickle_data(os.path.join(baseDir, seq, 'meta',  filename))

            try:
                _, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
            except:
                continue

            objMesh = read_obj(os.path.join(YCBModelsDir,'models', anno['objName'], 'textured_simple.obj'))
            objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
            objMesh_faces, o3d_obj_mesh = mesh2open3dMesh2npy([objMesh], ['r'], 'obj') 
            handMesh_faces, o3d_hand_mesh = mesh2open3dMesh2npy([handMesh], ['g'], 'hand')
            
            if not os.path.exists(os.path.join(baseDir, seq, 'mesh_hand')):
                os.makedirs(os.path.join(baseDir, seq, 'mesh_hand'))     
            if not os.path.exists(os.path.join(baseDir, seq, 'mesh_obj')):
                os.makedirs(os.path.join(baseDir, seq, 'mesh_obj'))     
            
            open3d.io.write_triangle_mesh(os.path.join(baseDir, seq, 'mesh_hand',
                os.path.splitext(filename)[0]+'.obj'), o3d_hand_mesh[0])
            open3d.io.write_triangle_mesh(os.path.join(baseDir, seq, 'mesh_obj',
                os.path.splitext(filename)[0]+'.obj'), o3d_obj_mesh[0])

    print("Mesh generation completed, save in folder:", os.path.join(baseDir, seq, 'meta', 'mesh'))
