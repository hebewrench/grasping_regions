from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.backend.sim import simGetIntegerSignal 
from pyrep.backend.sim import simSetIntegerSignal 
from pyrep.objects.object import Object
from pyrep.backend.sim import  simGetShapeMesh
from pyrep.backend.sim import  simGetObjectMatrix
from pyrep.backend.sim import  simGetObjectHandle
from pyrep.backend.sim import  simAddDrawingObject
from pyrep.backend.sim import  simAddDrawingObjectItem
from pyrep.backend.sim import simGetObjectHandle
from pyrep.backend.sim import simGetJointType
from pyrep.backend.sim import simGetJointPosition
from pyrep.backend.sim import simGetObjectPosition
from pyrep.backend.sim import simSetJointPosition
from pyrep.const import ConfigurationPathAlgorithms as Algos
#from pyrep.backend.sim import simSetThreadAutomaticSwitch
from pyrep.backend.simConst import sim_drawing_lines 
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
import numpy as np
from pyrep.const import PrimitiveShape
from numpy import  arange
from numpy.random import choice 
from pyrep.objects.vision_sensor import VisionSensor
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from random import randrange
#from pyrep.backend.sim import simGetObjectHandle 
import time
import math
from pyrep.robots.arms.panda import Panda
from pyrep.errors import ConfigurationPathError

def Ry(angle):
 return np.array([[np.cos(angle), 0, np.sin(angle)],
                 [0, 1, 0], 
                 [-np.sin(angle), 0, np.cos(angle)]])
def Rz(angle):
 return np.array([[np.cos(angle), -np.sin(angle), 0,0],
                 [np.sin(angle), np.cos(angle), 0,0], 
                 [0, 0, 1,0],
                 [0, 0, 0,1]])
def Rx(angle):
 return np.array([[1, 0, 0],
                 [0, np.cos(angle), -np.sin(angle)], 
                 [0, np.sin(angle), np.cos(angle)]])

def orientation(normal):
 a=np.array([[0],[0],[1]])
 normal=-np.array([[-1],[1],[-1]])*normal
 v= np.cross(a,normal,axis=0)
 c= np.dot(a.flatten(),normal.flatten())
 s=np.linalg.norm(v)
 v_skew = np.array( [ [0,-v[2][0],v[1][0]], [v[2][0],0,-v[0][0]], [-v[1][0],v[0][0],0] ] ) 
 R= np.identity(3)+ v_skew + np.dot(v_skew,v_skew)*( (1-c)/(s*s))
 return R

def HTM(R,p):
 return np.vstack( ( np.hstack( (R,p) ),np.array([0,0,0,1]) ) )
 

def config_HTM(n,ancho): #n=[x1,x2,x3,normalx,normaly,normalz]
 desp_z= np.array([[1,0,0,0],[0,1,0,0],[0,0,1,(n[2]/n[2])*(ancho/2)],[0,0,0,1]])
 #-------------------------------------------------------
 r= orientation( np.array(n[3:]).reshape(-1,1)  )
 pos = np.array([[-n[0]*1000],[n[1]*1000],[-n[2]*1000]])
 dum_test = np.hstack( (r,pos) )
 #print("dum_test",dum_test)
 dum_test = np.matmul( np.matmul(dum_test, desp_z ), Rz(-np.pi/2) )
 #dum_test = np.matmul( dum_test, Rz(-np.pi/2) )
 #print("dum: ", dum_test)
 return dum_test

def par_cir(r,axis,a,b):
  circ=[]
  for i in range(314*2):
    if axis=="x":
      x=0
      y=r*np.cos(i/100)+a
      z=r*np.sin(i/100)+b
    if axis=="y":
      x=r*np.cos(i/100)+a
      y=0
      z=r*np.sin(i/100)+b
    if axis=="z":
      x=r*np.cos(i/100)+a
      y=r*np.sin(i/100)+b
      z=0
    
    circ.append([x,y,z])
  return circ

def cylin(start,end,axis,a,b):
  cylin=np.array([[0,0,0]])
  for i in np.arange(start,end,0.5):
    circ = np.array(par_cir(1,axis,a,b))
    if axis=="x":
     circ[:,0] = i
    elif axis=="y":
     circ[:,1] = i
    elif axis=="z":
     circ[:,2] = i
    cylin=np.vstack((cylin,circ))
  cylin=np.delete(cylin,0,axis=0)
  
  return cylin

def gripper(w,mango_start,mango_end,mango_dy,mango_dz):
  '''Funciona para XY y YX
  mango=cyl(mango_start,mango_end+3,"z",mango_dy,mango_dz)
  base=cyl(-w/2+mango_dy-4,w/2+mango_dy+4,"x",mango_end+2,0)
  finger_1=cyl(mango_end,mango_end+16,"z",mango_dy+(w/2)+2,0)
  finger_2=cyl(mango_end,mango_end+16,"z",mango_dy-(w/2)-2,0)
  '''
  mango=cylin(mango_start,mango_end+1.5,"z",mango_dy,mango_dz)
  #print("mango: ", mango)
  base=cylin(-w/2+mango_dy-2,w/2+mango_dy+2,"x",0,mango_end+1)
  finger_1=cylin(mango_end,mango_end+40,"z",mango_dy+(w/2)+1,0)
  finger_2=cylin(mango_end,mango_end+40,"z",mango_dy-(w/2)-1,0)

  grip=np.vstack( (mango,base ,finger_1,finger_2,np.array([[0,0,0]]) ) )/-1000000
  return grip

def rotate_gripper(R,p):
 HTM_final=[]
 HTM_1=np.vstack( (np.hstack((R,p)), np.array([[0,0,0,1]])) )

 grip_mod=gripper(49,-55,-35,0,0)
 #grip_pcd = o3d.geometry.PointCloud()
 #grip_pcd.points = o3d.utility.Vector3dVector(grip_mod)
 #print(np.full((len(grip_pcd.points), 3), [1,0,0]))
 #color=np.full((len(grip_pcd.points), 3), [1,0,0])
 #grip_pcd.colors= o3d.utility.Vector3dVector(color)
 ch=0
 
 for i in grip_mod:
  ch=ch+1
  #print(ch)
  #print(len(grip_mod))
  HTM_final.append(np.matmul(HTM_1 ,np.array([[1,0,0,i[0]],[0,1,0,i[1]],[0,0,1,i[2]],[0,0,0,1]]) )[:3,3] )
  #print(np.matmul(HTM_1 ,np.array([[1,0,0,i[0]],[0,1,0,i[1]],[0,0,1,i[2]],[0,0,0,1]]) )[:3,3])
 
 return np.array(HTM_final)
 


grip_model=gripper(48,-55,-35,0,0)
grip_pcd = o3d.geometry.PointCloud()
grip_pcd.points = o3d.utility.Vector3dVector(grip_model)
#print( "model:",grip_model ) #
#print(np.full((len(grip_pcd.points), 3), [1,0,0])) #
color=np.full((len(grip_pcd.points), 3), [1,0,0])
grip_pcd.colors= o3d.utility.Vector3dVector(color)

def vision():
   id_pc=[]
   start_time = time.time()
   vision = VisionSensor("Vision_sensor")
   vision.set_explicit_handling(value=1)
   vision.handle_explicitly()
   depth=vision.capture_depth(in_meters=True)
   rgb=vision.capture_rgb()

   img_depth = o3d.geometry.Image((depth*255).astype(np.uint8))
   #o3d.io.write_image("images/SRd.png", img_depth)
   #o3d.visualization.draw_geometries([img_depth])
   img_rgb = o3d.geometry.Image((rgb*255).astype(np.uint8))
   #o3d.io.write_image("images/SRc.png", img_rgb)
  # o3d.visualization.draw_geometries([img_rgb])

   rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_rgb, img_depth,convert_rgb_to_intensity=False)
   w=vision.get_resolution()[0]
   h=vision.get_resolution()[1]
   fl_x=vision.get_intrinsic_matrix()[0][0]
   fl_y=vision.get_intrinsic_matrix()[1][1]
   c_x=vision.get_intrinsic_matrix()[0][2]
   c_y=vision.get_intrinsic_matrix()[1][2]

   intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h,fx=fl_x,fy=fl_y,cx=c_x,cy=c_y)
   pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
   pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   #o3d.visualization.draw_geometries([pcd])
  # print(np.asarray(pcd.points)) 
   downpcd = pcd.voxel_down_sample(voxel_size=0.000005) #voxel_size=0.000009

   # o3d.visualization.draw_geometries([downpcd])
  # print("color: ",downpcd.colors[500])
   #downpcd.colors[500]=[0,0,1]
   downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5))
   downpcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
   downpcd.normalize_normals()

   new_downpcd=[]
#  [0.84705882 0.84705882 0.84705882]

#Para este código, la mesa tiene un color de [0.77647059 0.72941176 0.69411765]
   for i in range(len(np.asarray(downpcd.colors))):
    sup_mesa_col= 0.78705882<=downpcd.colors[i][0]<=0.90705882 and 0.78705882<=downpcd.colors[i][1]<=0.90705882 and 0.78705882<=downpcd.colors[i][2]<=0.90705882
    neg= downpcd.colors[i][0]==0 and downpcd.colors[i][1]==0 and downpcd.colors[i][2]==0
    #lado_mesa_col= 0.40568627<=downpcd.colors[i][0]<=0.42568627 and 0.33117647<=downpcd.colors[i][1>
    #bord_mesa_col= 0.49862745<=downpcd.colors[i][0]<=0.55862745 and 0.43666667<=downpcd.colors[i][1>

    if (sup_mesa_col ): #or neg): #or (lado_mesa_col) or bord_mesa_col:
     print("encontrado",end='\r')
    else:
     new_downpcd.append([downpcd.points[i][0],downpcd.points[i][1],downpcd.points[i][2],downpcd.colors[i][0],downpcd.colors[i][1],downpcd.colors[i][2]])

  
  # #print("newpos:",np.array(new_downpcd))  

   downpcd = o3d.geometry.PointCloud()
   downpcd.points=o3d.cpu.pybind.utility.Vector3dVector( np.array(new_downpcd)[:,:3]  )
   downpcd.colors=o3d.cpu.pybind.utility.Vector3dVector( np.array(new_downpcd)[:,3:] )

#for i in range( len(np.asarray(downpcd.colors)) ):
 #print(i+114)
 #downpcd.colors[i+114]=[1,0,0]
   #print(np.asarray( downpcd.colors[114]))
   # o3d.visualization.draw_geometries([downpcd])
   downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5))
   downpcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
   downpcd.normalize_normals()


   #print(np.asarray(downpcd.normals)) 
   for i in np.asarray(downpcd.normals):
    np.linalg.norm(i)
   #o3d.visualization.draw_geometries([downpcd],point_show_normal=True)


   #transformar el downsampled pointcloud a puntos en np.array
   points = np.asarray(downpcd.points).copy()
   #points = np.asarray(pcd.points).copy(
   scaled_points = StandardScaler().fit_transform(points)

   #o3d.visualization.draw_geometries([downpcd])
   model = DBSCAN(eps=0.06, min_samples=10) #eps=0.15
   model.fit(scaled_points)
#visualizar
# Get labels:
   labels = model.labels_
# Get the number of colors:
   n_clusters = len(set(labels))
   #print(n_clusters)#
   #print(labels)#
   #print(scaled_points)#
# Mapping the labels classes to a color map:
   colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))

# Attribute to noise the black color:
   colors[labels < 0] = 0
# Update points colors:
   downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#print(colors[:, :3])

   tipos=np.array([[0,0,0]])

   for i in np.asarray(downpcd.colors)[:, :3]:
    if len(tipos)==0:
     tipos=np.vstack([tipos,i])
  #print("holai")
    rep=[]
    for j in tipos:
     rep.append((i!=j).any())
    if not (False in rep):
     tipos=np.vstack([tipos,i])
   
   #print("n_tipos:", len(tipos)) 
 
  
#print(tipos)
#print(tipos)
   #print("----------------------------")
# Display:
   #o3d.visualization.draw_geometries([downpcd])
   comp_arr=np.hstack([downpcd.points,downpcd.colors])
#---------------------------------Observación de todos los clusters---------------------------------
   list_nptos=[]
   for i in range(len(tipos)):
    arr_color= comp_arr[ (comp_arr[:,3]==tipos[i][0]) & (comp_arr[:,4]==tipos[i][1]) & (comp_arr[:,5]==tipos[i][2]) ][:,3:]
    #print("tipo: ", tipos[i] )
    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points=o3d.cpu.pybind.utility.Vector3dVector( comp_arr[ (comp_arr[:,3]==tipos[i][0]) & (comp_arr[:,5]==tipos[i][2]) ][:,:3] )
    seg_pcd.colors=o3d.cpu.pybind.utility.Vector3dVector( arr_color )
    #print("#puntos: ",len(np.asarray(seg_pcd.points)))
    if  230<len(np.asarray(seg_pcd.points))<400:
     list_nptos.append(i)

    seg_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=5))
    seg_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    seg_pcd.normalize_normals()

    for n in np.asarray(seg_pcd.normals):
     np.linalg.norm(n)
    # o3d.visualization.draw_geometries([seg_pcd],point_show_normal=True)
   arr_nptos=np.array(list_nptos) #indices de los objetos identificados
   #print( arr_nptos[(200<arr_nptos<300).all()] )
   #print(arr_nptos)
   #print(np.where(200<arr_nptos<400))
   ancho=0
   n_cil=0
   all_gripoint=[]
   for i in arr_nptos:
    arr_color= comp_arr[ (comp_arr[:,3]==tipos[i][0]) & (comp_arr[:,4]==tipos[i][1]) & (comp_arr[:,5]==tipos[i][2]) ][:,3:]

    cyl_pcd = o3d.geometry.PointCloud()
    cyl_pcd.points=o3d.cpu.pybind.utility.Vector3dVector( comp_arr[ (comp_arr[:,3]==tipos[i][0]) & (comp_arr[:,4]==tipos[i][1]) & (comp_arr[:,5]==tipos[i][2]) ][:,:3] )
    cyl_pcd.colors=o3d.cpu.pybind.utility.Vector3dVector( arr_color )
    cyl_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=5)) #radius=0.01
    cyl_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    cyl_pcd.normalize_normals()
    #print("-------------------ptos en cluster-----------------")
    #print("ptosclust: ", len(cyl_pcd.points) )
    #o3d.visualization.draw_geometries([cyl_pcd],point_show_normal=True)
    id_pc.append(cyl_pcd)
    end_time = time.time()
    execution_time = end_time - start_time
    print("total object detection time:", execution_time)
    start_time=time.time()
    cyl_points_xmax=np.max(np.asarray(cyl_pcd.points)[:,0])*1000
    cyl_points_xmin=np.min(np.asarray(cyl_pcd.points)[:,0])*1000
    cyl_points_zmax=np.max(np.asarray(cyl_pcd.points)[:,2])
    cyl_points_zmin=np.min(np.asarray(cyl_pcd.points)[:,2])
    cyl_points_ymax=np.max(np.asarray(cyl_pcd.points)[:,1])
    cyl_points_ymin=np.min(np.asarray(cyl_pcd.points)[:,1])
    
    mitad=(cyl_points_xmax+cyl_points_xmin)/(2*1000)
    mitad_z=(cyl_points_zmax+cyl_points_zmin)/(2)
    mitad_y=(cyl_points_ymax+cyl_points_ymin)/(2)
    
    #---------------------------Agregar gripper------------------------------------
    #print("cyl_p: ",cyl_pcd.points[50][0],cyl_pcd.points[50][1],cyl_pcd.points[50][2])
    r_gripper=rotate_gripper(Rx(- (0 *(np.pi/180))),np.array([[mitad],[mitad_y],[mitad_z]]))
    all_gripoint.append(r_gripper)

    points_comp = np.concatenate((np.asarray(cyl_pcd.points), r_gripper))
    colors_comp = np.concatenate((np.asarray(cyl_pcd.colors), grip_pcd.colors))
    pcd_comp = o3d.geometry.PointCloud()
    pcd_comp.points = o3d.utility.Vector3dVector(points_comp)
    pcd_comp.colors = o3d.utility.Vector3dVector(colors_comp)
    #o3d.visualization.draw_geometries([pcd_comp])
    #------------------------------------------------------------------------------

    ancho = ancho + (cyl_points_xmax-cyl_points_xmin)
    n_cil= n_cil+1
    points_axis.append( [ cyl_pcd.points[50][0], cyl_pcd.points[50][1], cyl_pcd.points[50][2], cyl_pcd.normals[50][0], cyl_pcd.normals[50][1], cyl_pcd.normals[50][2] ])
    cyl_pcd.colors[100]=[0,0,1]
   ancho= (ancho/n_cil) #*1000
   #print("ancho_med: ", ancho)

   p=0
   #o3d.visualization.draw_geometries([pcd])
   grip_scpoints=np.asarray(pcd.points)
   grip_sccolors=np.asarray(pcd.colors)
   for i in all_gripoint:
    grip_scpoints=np.vstack((grip_scpoints, i))
    grip_sccolors=np.vstack((grip_sccolors, np.full((len(i), 3), [1,0,0])))
   gripsc_comp = o3d.geometry.PointCloud()
   gripsc_comp.points = o3d.utility.Vector3dVector(grip_scpoints)
   gripsc_comp.colors = o3d.utility.Vector3dVector(grip_sccolors)
   #o3d.visualization.draw_geometries([gripsc_comp])
   end_time = time.time()
   execution_time = end_time - start_time
   print("Grasp Pose detection time:",execution_time)
   return id_pc, ancho

LOOPS=1

#print(Ry(np.pi))
pr = PyRep()
pr.launch('tesispy_one.ttt', headless=False) 
pr.start()
panda = Panda()
#print(panda._collision_collection)
frankaHand=PandaGripper(count= 0)
Dummy.create(size=0.1).set_matrix( HTM(Ry(np.pi/2), np.array([[0.1],[0.1],[0.1]]) )  )
Dummy0= Object.get_object("Dummy0")
#print("Dummy= ",Dummy0.get_matrix(relative_to=None))
Dummy0.set_matrix( HTM( np.matmul( np.matmul( Ry(np.pi/2), Ry( 5*(np.pi/180) )[:3,:3] ), Rz( 5*(np.pi/180) )[:3,:3]  ), np.array([[0.1],[0.1],[0.1]]) )  )
#print("Dummy= ",Dummy0.get_matrix(relative_to=None))
#_______________________________Unloading poses________________________________-______
goals=[] #end effector goals
targetDummy= Object.get_object("target1")
table= Object.get_object("customizableTable_tableTop0")
table_vertices=table.get_bounding_box()  
table_pos=table.get_position(relative_to=None)
uldp=[] #unloading poses
cyl_size=[0.048,0.048,0.1]
space=0.03
num_cylx=(abs(-table.get_bounding_box()[0]+table.get_bounding_box()[1]))/(cyl_size[0]+space*2)
num_cyly=(abs(-table.get_bounding_box()[2]+table.get_bounding_box()[3]))/(cyl_size[1]+space*2)

#print(int(num_cylx))

cyl_pos= [table_pos[0]+table_vertices[1]+(cyl_size[0]/2)+space,table_pos[1]-table_vertices[3]+(cyl_size[0]/2)+space, table_pos[2]+0.04]
cyl_pos[-1]=table_pos[2]+(cyl_size[2]/2) #+0.0045) # 0.04
values=arange(0, 1, 0.1)

for j in list(range(5)):
 for i in list(range(5)):
  #print(i)
  
  cyl_pos[0]= cyl_pos[0]-(space*2+cyl_size[0])
  uldp.append([cyl_pos[0],cyl_pos[1],cyl_pos[2]])


  if i==int(round(num_cyly))-1:
   cyl_pos[0]= table_pos[0]+table_vertices[1]+(cyl_size[0]/2)+space
 
 cyl_pos[1]= cyl_pos[1]+space*2+cyl_size[1]


uldp[4]=[uldp[3][0],uldp[3][1]+0.08,cyl_pos[2]]
#___________________________________________________________________________________________
#print("starting positions:")
starting_joint_positions = panda.get_joint_positions()
#print("target:")
target= Object.get_object("target1")
cyls=[Shape("Cylinder")]
#print("vertices: ",cyls[0].get_bounding_box() )
base=cyls[0].get_position()[2]+cyls[0].get_bounding_box()[4]
#print("base: ",base)
obs_pos=target.get_position()
obs_quat=target.get_quaternion(relative_to=None)
obs_mat=target.get_matrix(relative_to=None)
panda.set_joint_positions(starting_joint_positions)
#print("starting joint positions")

for i in range(LOOPS):
 for ncyl in range(len(cyls)):
    #print(ncyl)
    points_axis=[]
    # Reset the arm at the start of each 'episode'
    
    # Get a random position within a cuboid and set the target position
    #pos = list(np.random.uniform(position_min, position_max))
    


    #---------------------Observation pose--------------------
    # Get a path to the target (rotate so z points down)
    target.set_matrix(obs_mat, relative_to=None)
    try:
        path = panda.get_path(
            position=obs_pos, quaternion=obs_quat, algorithm=Algos.RRTConnect, max_configs=5, trials=600)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = path.step()
        pr.step()
    
    #print('Reached target %d!' % i)
    #---------------------------Vision and filter----------------------------------
   
    id_cyl,width=vision() #identified cylinders
    start_time = time.time()
    pos_rbasel=[] #position (z) wrt the base
    for i in range(len(id_cyl)):
     #print("pts= ",np.asarray(id_cyl[i].points))
     #print("normals= ",np.asarray(id_cyl[i].normals))
     n_list=np.hstack( (np.asarray(id_cyl[i].points),np.asarray(id_cyl[i].normals)) )
     #print(len(n_list))
     cyl_dum=[] #lista para un cilindro identificado que tenga el eje x apuntando hacia abajo y sin colisión
     cyl_path=[]
     cyl_plength=[] 
     for j in  range(len(n_list)):
      #print(n_list[j])
      htm_test=config_HTM(n_list[j],0.048) #n=[x1,x2,x3,normalx,normaly,normalz]
      dum=Dummy.create(size=0.01).get_handle()
      Dummy_object= Dummy(dum)
      Dummy_object.set_matrix(htm_test, relative_to=VisionSensor("Vision_sensor"))
      dumHTMx=Dummy_object.get_matrix(relative_to=None)[:3,0]
      #dum.get_matrix(relative_to=None)
      '''
      print("-----------------------------------")
      print(Dummy_object.get_matrix(relative_to=None))
      print(dumHTMx)
      print(-0.087<=dumHTMx[0]<=0.087)
      print(-0.087<=dumHTMx[1]<=0.087)
      print(-1<=dumHTMx[2]<=-0.997)
      print("-----------------------------------")
      '''
      orient_cond=-0.009<=dumHTMx[0]<=0.009 and -0.009<=dumHTMx[1]<=0.009 and -1<=dumHTMx[2]<=-0.999
      #orient_cond=-0.087<=dumHTMx[0]<=0.087 and -0.087<=dumHTMx[1]<=0.087 and -1<=dumHTMx[2]<=-0.97
      if orient_cond:
       
       #print("Dummy_world= ", Dummy_object.get_matrix(relative_to=None) )
       try:
        #configs=panda.get_configs_for_tip_pose(position=Dummy_object.get_position(), quaternion=Dummy_object.get_quaternion() )
        path = panda.get_path(
            position=Dummy_object.get_position(), quaternion=Dummy_object.get_quaternion(), algorithm=Algos.RRTConnect, max_configs=5, trials=600)
        #print("----------------------")
        path.visualize()
        #print("path_length:",path._get_path_point_lengths()[-1])
        pr.step()
        
        #print(  path._path_points  )
        #print("----------------------")
        if len(path._path_points)>0:
         cyl_dum.append(Dummy_object)
         cyl_path.append(path)
         cyl_plength.append(path._get_path_point_lengths()[-1])
        
       except:
        #print("no se encontró")
        Dummy_object.remove()
       path.clear_visualization()  
       
      else:
       #print("hola")
       Dummy_object.remove()
     chosen_path=np.argmin(cyl_plength)
     print("chosen path: ",print(chosen_path))
     print("chosen_length: ", cyl_plength[chosen_path])
     print("pos_rbase: ", abs(cyl_dum[chosen_path].get_matrix()[2,3]-base))
     pos_rbasel.append(abs(cyl_dum[chosen_path].get_matrix()[2,3]-base))
     cyl_path[chosen_path].visualize()
     pr.step()
     print("or_val=",len(cyl_dum))  
     end_time = time.time()
     execution_time = end_time - start_time
     print("motion planning:",execution_time)
     start_time= time.time()
    #---------------------------Grasping pose----------------------------
    cyls[ncyl].get_position()
    #print('rot_bef= ',cyls[ncyl].get_matrix(relative_to=None)[0:3,0:3])
    rot=np.matmul( np.matmul( cyls[ncyl].get_matrix(relative_to=None)[0:3,0:3],Rz(np.pi/2)[0:3,0:3] ),
    Ry(np.pi/2) )
    #print('rot_aft= ',rot)

    goalm=cyls[ncyl].get_matrix(relative_to=None)
    goalm[0:3,0:3]=rot
    target.set_matrix(goalm, relative_to=None)
    for i in range(100):
     pr.step()
    '''
    try:
        path = panda.get_path(position=target.get_position(), quaternion=target.get_quaternion(relative_to=None), algorithm=Algos.RRTConnect, max_configs=5, trials=600)
        path.visualize()
        pr.step()
    except ConfigurationPathError as e:
        print('Could not find path')
        continue
    '''

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = cyl_path[chosen_path].step()
        pr.step()
    #print('Reached target %d!' % i)
    #--------------------------Grasp-------------------------------------
    k=0
    while (k !=300):
     frankaHand.actuate(amount= 0, velocity= 0.005)
     #print(i)
     k=k+1
     pr.step()
    cyl_g= cyls[ncyl]
    cyl_g.set_collidable(False)
    #frankaHand.grasp(cyl_g)
    frankaHand.grasp(cyl_g)
   #-------------------------Unloading pose------------------------------
    
    #print('rot_bef= ',cyls[ncyl].get_matrix(relative_to=None)[0:3,0:3])
    rot= Ry(np.pi/2)
    #print('rot_aft= ',rot)

    #print('pos= ',np.array([uldp[ncyl]]).T)
    goalm=cyl_dum[chosen_path].get_matrix(relative_to=None)
    #print(goalm)
    goalm[0:3,0:3]=rot
    #print(goalm[:3,3])
    #cyl_dum[chosen_path]
    goalm[:3,3]=np.array(uldp[ncyl])
    goalm[2,3]=pos_rbasel[0]+table_pos[2]
    #print(goalm)
    target.set_matrix(goalm, relative_to=None)

    
    for i in range(100):
     pr.step()
    
    try:
        path = panda.get_path(
            position=target.get_position(), quaternion=target.get_quaternion(relative_to=None),algorithm=Algos.RRTConnect, max_configs=5, trials=600)
    except ConfigurationPathError as e:
        print('Could not find path')
        continue
    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = path.step()
        pr.step()
    #print('Reached target %d!' % i)
   #-------------------------Release------------------------------
    k=0
    while (k !=300):
     frankaHand.actuate(amount= 1, velocity= 0.005)
     #print(i)
     k=k+1
     pr.step()
  
    cyl_g= cyls[ncyl]
    cyl_g.set_collidable(True)
    frankaHand.release()

    end_time = time.time()
    execution_time = end_time - start_time
    print("motion execution time:",execution_time)

for l in range(900):
 pr.step()
pr.stop()
pr.shutdown()
