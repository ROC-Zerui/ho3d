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

# # mapping of joints from MANO model order to simple order(thumb to pinky finger)
# jointsMapManoToSimple = [0,
#                          13, 14, 15, 16,
#                          1, 2, 3, 17,
#                          4, 5, 6, 18,
#                          10, 11, 12, 19,
#                          7, 8, 9, 20]

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

    #print(m.fullpose.shape)

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
            # raw_input("enter for watertight triangles")
            # print(mano_watertight_mesh.triangles)
            # raw_input("enter to continue")
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


def mean_center_o3d_mesh(o3d_mesh_comb, o3d_mesh_hand, o3d_mesh_obj, color=None):
    # hand mesh vertices mean
    o3d_mesh_hand_mean_v = np.asarray(np.mean(o3d_mesh_hand.vertices, axis=0)).reshape((1, -1))
    
    # hand mesh vertices mean
    o3d_mesh_obj_mean_v = np.asarray(np.mean(o3d_mesh_obj.vertices, axis=0)).reshape((1, -1))
    # raw_input("enter for hand and obj  v mean")
    print("o3d_mesh_hand_mean_v", o3d_mesh_hand_mean_v)
    print("o3d_mesh_obj_mean_v", o3d_mesh_obj_mean_v)
    
    # hand mesh weight for weighted avg
    print("np.asarray(o3d_mesh_hand.vertices).shape[0]", len(o3d_mesh_hand.vertices))
    print("np.asarray(o3d_mesh_obj.vertices).shape[0]", len(o3d_mesh_obj.vertices))
    o3d_mesh_hand_w = float(len(o3d_mesh_hand.vertices)) / float((len(o3d_mesh_hand.vertices) + len(o3d_mesh_obj.vertices)))
    
    # obj mesh weight for weighted avg
    o3d_mesh_obj_w = float(len(o3d_mesh_obj.vertices)) / float((len(o3d_mesh_hand.vertices) + len(o3d_mesh_obj.vertices)))
    
    # raw_input("enter for weight")
    print("o3d_mesh_hand_w", o3d_mesh_hand_w)
    print("o3d_mesh_obj_w", o3d_mesh_obj_w)

    # weighted average of combined mesh vertices 
    # (this is equivalent to averaging the combined mesh vertices)
    # not used!!(non-weighted average is used as we prefer center to middle instead of towards the mesh with more vertices)
    weighted_vertices_mean = (o3d_mesh_hand_w *  o3d_mesh_hand_mean_v) + (o3d_mesh_obj_w * o3d_mesh_obj_mean_v)
    # raw_input("enter for weighted_vertices_mean")
    print("weighted_vertices_mean", weighted_vertices_mean)
    centered_mesh = open3d.geometry.TriangleMesh()
    
    # Non-weighted average
    vertices_mean = (o3d_mesh_hand_mean_v + o3d_mesh_obj_mean_v) / 2.0
    # raw_input("enter for non-weighted vertices_mean")
    print("Non weighted_vertices_mean", vertices_mean)
    # raw_input("wait and check")

    # combined mesh mean centering using weighted avg mean
    centered_mesh.vertices = open3d.utility.Vector3dVector(np.copy(np.subtract(np.asarray(o3d_mesh_comb.vertices), vertices_mean)))  
    centered_mesh.triangles = open3d.utility.Vector3iVector(np.copy(np.asarray(o3d_mesh_comb.triangles)))
    numVert = np.asarray(o3d_mesh_comb.vertices).shape[0]

    if color is 'red':
        centered_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.6, 0.2, 0.2]]), [numVert, 1]))
    elif color is 'grey':
        centered_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))
    else:
        # default mesh color
        print("mesh color not specified, default mesh color used")
        centered_mesh.vertex_colors = open3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), [numVert, 1]))

    return centered_mesh


if __name__ == '__main__':
    # parse the input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-meta_data_path", type=str, default="/home2/zerui/dataset/HO3D_V3/train",
    help="Path to HO3D dataset train or val or test directory", required=False)
    ap.add_argument("-ycbModels_path", type=str, default="/home2/zerui/dataset/HO3D_V3/modelsprocess",
    help="Path to ycb models directory", required=False)
    ap.add_argument("-bbox_extent_min", type=float, default=-0.25, 
    help="min bound the square bbox", required=False)
    ap.add_argument("-bbox_extent_max", type=float, default=0.25, 
    help="max bound the square bbox", required=False)

    args = vars(ap.parse_args())

    baseDir = args['meta_data_path'] 
    YCBModelsDir = args['ycbModels_path']

    # create min and max bound for cube
    bbox_min_bound = np.array([args['bbox_extent_min']]*3)
    bbox_max_bound = np.array([args['bbox_extent_max']]*3)

    # diameter of sphere enclosing bbox(dia=cuberoot(3)*bbox_length)
    BBOX_DIAMETER = np.sqrt(3) * (bbox_max_bound-bbox_min_bound)

    print(args)
    seq_no = 0
    for seq in os.listdir(baseDir):
        # print(os.path.join(baseDir, seq, 'meta'))
        for filename in os.listdir(os.path.join(baseDir, seq, 'meta')):
            print("filepath:", os.path.join(baseDir, seq, 'meta',  filename))
            anno = load_pickle_data(os.path.join(baseDir, seq, 'meta',  filename))

            try:
                _, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
            except:
                continue

	    #print(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))
            objMesh = read_obj(os.path.join(YCBModelsDir,'models', anno['objName'], 'textured_simple.obj'))
            objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
            objMesh_faces, o3d_obj_mesh = mesh2open3dMesh2npy([objMesh], ['r'], 'obj') 
            handMesh_faces, o3d_hand_mesh = mesh2open3dMesh2npy([handMesh], ['g'], 'hand')
             
            # raw_input("enter for hand and obj mesh before mean centering")
            # open3d.visualization.draw_geometries([o3d_hand_mesh[0], o3d_obj_mesh[0]])

            ## meshes mean centering
            # o3d_hand_mesh_centered = mean_center_o3d_mesh(o3d_hand_mesh[0], color='red')
            # o3d_obj_mesh_centered = mean_center_o3d_mesh(o3d_obj_mesh[0], color='grey')
            # raw_input("enter for hand and obj mesh AFTER  mean centering")            
            # print(np.mean(np.asarray(o3d_hand_mesh_centered.vertices)))
            # print(np.mean(np.asarray(o3d_obj_mesh_centered.vertices)))
            # open3d.visualization.draw_geometries([o3d_hand_mesh_centered, o3d_obj_mesh_centered])
            
            ## combined mesh centering
            hand_obj_mesh_comb = o3d_hand_mesh[0] + o3d_obj_mesh[0]
            # print(hand_obj_mesh_comb, o3d_hand_mesh[0], o3d_obj_mesh[0])
            # raw_input("enter for hand and obj combined mesh BEFORE mean centering")
            # open3d.visualization.draw_geometries([hand_obj_mesh_comb])
            hand_obj_mesh_comb_centered = mean_center_o3d_mesh(hand_obj_mesh_comb, o3d_hand_mesh[0], o3d_obj_mesh[0], color='red')
            print("centered mesh V mean:", np.mean(np.asarray(hand_obj_mesh_comb_centered.vertices)))
            # raw_input("enter for hand and obj combined mesh AFTER mean centering")
            # open3d.visualization.draw_geometries([hand_obj_mesh_comb_centered])

            ## Bounding box block
            # Axis aligned BBox creation
            aabb = open3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min_bound, max_bound=bbox_max_bound)
            print("default aabb vol:", hand_obj_mesh_comb_centered.get_axis_aligned_bounding_box().volume())
            # open3d.visualization.draw_geometries([hand_obj_mesh_comb_centered, aabb])
            print("created aabb vol:", aabb.volume())
            # raw_input("Next is scaling by DIA")

            ## Scale the vertices by diameter/diagonal of the bounding box
            # hand_obj_mesh_comb_centered_vertices_scaled = np.divide(hand_obj_mesh_comb_centered.vertices, BBOX_DIAMETER)
            print(np.asarray(hand_obj_mesh_comb_centered.vertices))
            hand_obj_mesh_comb_centered.vertices = open3d.utility.Vector3dVector(np.divide(np.asarray(hand_obj_mesh_comb_centered.vertices), BBOX_DIAMETER))
            print(np.asarray(hand_obj_mesh_comb_centered.triangles), np.asarray(hand_obj_mesh_comb_centered.triangles).shape)
            hand_obj_mesh_comb_centered_triangles = np.asarray(hand_obj_mesh_comb_centered.vertices)[np.asarray(hand_obj_mesh_comb_centered.triangles)]
            
            # open3d.visualization.draw_geometries([hand_obj_mesh_comb_centered, aabb])

            # if False:
            # writing section
            # if not os.path.exists(os.path.join(baseDir, seq, 'mesh_watertight')):
            #     os.makedirs(os.path.join(baseDir, seq, 'mesh_watertight')) 

            if not os.path.exists(os.path.join(baseDir, seq, 'hand_obj_mesh_watertight')):
                os.makedirs(os.path.join(baseDir, seq, 'hand_obj_mesh_watertight'))     
            
            # if not os.path.exists(os.path.join(baseDir, seq, 'mesh_watertight', 'hand')):
            #     os.makedirs(os.path.join(baseDir, seq, 'mesh_watertight', 'hand'))

            # with open(os.path.join(baseDir, seq, 'mesh_watertight', 'hand', os.path.splitext(filename)[0]+'.npy'), 'wb') as f:
            #     np.save(f, handMesh_faces[0])

            with open(os.path.join(baseDir, seq, 'hand_obj_mesh_watertight', os.path.splitext(filename)[0]+'.npy'), 'wb') as f:
                np.save(f, hand_obj_mesh_comb_centered_triangles)
            
            open3d.io.write_triangle_mesh(os.path.join(baseDir, seq, 'hand_obj_mesh_watertight',
                os.path.splitext(filename)[0]+'.obj'), hand_obj_mesh_comb_centered)

                # if not os.path.exists(os.path.join(baseDir, seq, 'mesh', 'obj')):
                #     os.makedirs(os.path.join(baseDir, seq, 'mesh', 'obj'))
                # with open(os.path.join(baseDir, seq, 'mesh', 'obj',  os.path.splitext(filename)[0]+'_mesh_obj.npy'), 'wb') as f:
                #     np.save(f, objMesh_faces[0])

	
	#seq_no += 1
        #print(seq_no, seq)
	#raw_input("enter for next sequence")
	
    print("Mesh generation completed, save in folder:", os.path.join(baseDir, seq, 'meta', 'mesh'))
