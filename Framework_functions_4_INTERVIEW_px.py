#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import load_model
import pdb
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as plt2
import math
import cv2
import pandas as pd
from numpy import save, asarray, linspace
from numpy.random import randn, randint
import tensorflow as tf
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os, glob, shutil, math
from os import listdir
from PIL import Image as PImage
from os.path import isfile,join
import scipy.io as sio
from scipy.spatial.distance import euclidean
#import matlab.engine as eng
#interactive plot
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image
import base64
from skimage.transform import resize
from skimage import measure
import plotly.express as px
from stl import mesh
import copy
from datetime import datetime
import inspect
import time

# ADD FUNCTION NAMES HERE THAT YOU DON'T WANT TO PRINT or add "all" if you don't want to print ANY functions
not_print = ['make_figure', 'compute_dimensions', 'frameworkCalcs'] # sv save edit
 
#FUNCTIONS MODIFIED FOR IDETC-2 FRAMEWORK 

def print_function(function_name): # sv save edit
  
    if function_name in not_print or 'all' in not_print:
        pass
    else:
        print(f'function: {function_name}')

# new saved img functions sv save edit
# def delete_imgs(delete_img_no, saved_imgs, sidebar_len)
def update_saved_imgs(saved_img_no,saved_imgs,sidebar_len): # sv save edit
    
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    if saved_imgs == [-1]:
        saved_imgs = []

    if saved_img_no in saved_imgs: # duplicate prevention
        feedback_msg = f"Design #{saved_img_no} has already been saved"
    else:
        #saved_imgs.append(saved_img_no)
        saved_imgs.insert(0, saved_img_no)
        feedback_msg = f"Design #{saved_img_no} has been saved"
   

    return saved_imgs, feedback_msg

def update_all_data(all_data, current_image, current_lv_data, current_cond_data, current_cost, current_dimensions):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    if np.shape(current_image) != (1,128,96,1):
        current_image = np.reshape(current_image, (1,128,96,1))
        
    all_data['lv_data'] = np.vstack((all_data['lv_data'], current_lv_data))
    all_data['cond_inputs'] = np.vstack((all_data['cond_inputs'], current_cond_data)) # NEW!
    all_data['cost'].append(current_cost)
  

    all_data['px_data'] = np.vstack((all_data['px_data'], current_image))
    all_data['dims'].append(current_dimensions)
    return all_data

def update_dropdown_options(saved_imgs):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    imgs = saved_imgs.copy()
    #imgs.reverse() # to show most recently saved imgs first
    print(f'saved_imgs: {imgs}')
    dropdown_options = [{'label': f'Saved Image #{len(saved_imgs) - i}', 'value': saved_imgs[i]} for i in range(len(saved_imgs))]
  
    return dropdown_options

def gen_sidebar_output(saved_imgs,sidebar_len,graphs,costs):
    # sidebar sync function: generates output for callback
    # insert graphs[saved_imgs] if saved_imgs !=-1 else 0
    # then skip to for i in range(..) & rest empty 
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    imgs = saved_imgs.copy()
    #imgs.reverse() # to show most recently saved images first
    print(f'saved_imgs: {imgs}')
    
    empty_img = {
                    'data': [{'type': 'image', 'z': np.zeros((256, 256, 3), dtype=np.uint8)}],
                    'layout':{'margin': '0px', 'padding': '0px', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}}}
    
    img_costs = [costs[i] for i in imgs]
    if saved_imgs == [-1]:
        num_saved_imgs = 0
    else:
        num_saved_imgs = len(saved_imgs)
        
    sidebar_outputs = [empty_img.copy() for i in range(sidebar_len)]
    sidebar_labels = [f'Saved Image {j+1}: Empty' for j in range(sidebar_len)]
    
    if num_saved_imgs > 0:
        if num_saved_imgs > sidebar_len:
            num_saved_imgs = sidebar_len
        for i in range(num_saved_imgs):
            sidebar_outputs[i] = graphs[imgs[i]].figure
            sidebar_labels[i] = f'Saved Image {len(saved_imgs)-i}, ${img_costs[i]}'
    
    return sidebar_outputs+sidebar_labels
    

def generate_imgs(num_images,noise,cond_inputs, gen_mdl):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    gen_imgs = gen_mdl.predict([noise,cond_inputs])
    #gen_imgs=gen_mdl([noise,vfs])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    return noise, gen_imgs





def gen_comp_vals(noise,eng):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    #sio.savemat('latent_vars.mat', {'latent_vars':noise})
    #eng.Kriging_to_python_4(nargout=0)
    #comp_vals = eng.workspace['comp_vals']
    #comp_scores=comp_vals[0]
    comp_scores = [0] * len(noise)
  
    return comp_scores

def gen_comp_val_slider(noise,eng):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    #no percentiles here
    
    #sio.savemat('latent_vars.mat', {'latent_vars':noise})
    #eng.Kriging_to_python_4(nargout=0)
    #comp_score = eng.workspace['comp_vals']
    comp_score = 0
    return comp_score

def interpolate_points(p1, p2, n_steps=10): # use this for blend & interp_points for suggest/pixclick
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)[:, np.newaxis]  # Create a column vector
    # linear interpolate vectors
    vectors = (1 - ratios) * p1 + ratios * p2
    return vectors
    

def cluster_imgs(X_gen, num_clusters):  
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    #  convert generated images to arrays
    #input_dir = '/Users/User/Documents/UCSDproject/Interactive Framework/Framework_gen_imgs'
    #glob_dir = input_dir + '/*.jpg'
    #images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    #paths = [file for file in glob.glob(glob_dir)]
    #images = np.array(np.float32(images).reshape(len(images), -1)/255)
    
    # convert image to correct shape for kmeans
    X_kmeans = resize(X_gen, (X_gen.shape[0], 224, 224, 3), anti_aliasing=True)
    X_kmeans = 1 - X_kmeans # black topology, white background
   

    #  Find features in the generated images using MobileNetV2
    #model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
    #predictions = model.predict(images.reshape(-1, 224, 224, 3))
    #pred_images = predictions.reshape(images.shape[0], -1)
    
    # apply feature extractor to images (consider applying PCA first to save time/space & see difference in end results)
    model = tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet', input_shape=(224, 224, 3)) # shouldn't this be 1, not 3?
    mob_predictions = model.predict(X_kmeans) # before: X_kmeans.reshape(-1,224,224,3) but now already in that shape
    pred_images = mob_predictions.reshape(X_kmeans.shape[0], -1)
    
    # apply kmeans
    kmodel = KMeans(n_clusters = num_clusters,  random_state=728)
    kmodel.fit(pred_images) #removed n_jobs=-1 b/c no longer kmeans feature (1/14/22)
    kpredictions = kmodel.predict(pred_images)
    
    return kpredictions, kmodel, mob_predictions # avg img edit

def find_avg_imgs(num_clusters,kpredictions,gen_imgs, noise, cond_inputs, gen_model): # avg_img edit
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)

    avg_imgs = np.empty((num_clusters, 128, 96, 1))
    for cluster in range(num_clusters):
        cluster_indices = np.where(kpredictions == cluster)[0]
        cluster_noise = noise[cluster_indices]
        avg_noise = np.mean(cluster_noise, axis=0, keepdims=True)
        cluster_cond_inputs = cond_inputs[cluster_indices]
        avg_cond_inputs = np.mean(cluster_cond_inputs, axis=0, keepdims=True)
        avg_img = gen_model.predict([avg_noise, avg_cond_inputs])
        avg_img = 0.5 * avg_img + 0.5
        avg_imgs[cluster] = avg_img

    return  avg_imgs

# def embeddable_image(data):
    
#     img_data = data.astype(np.uint8)
#     image = Image.fromarray(img_data, mode='L').resize((64, 32), Image.BICUBIC)
#     buffer = BytesIO()
#     image.save(buffer, format='png')
#     for_encoding = buffer.getvalue()
#     return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def make_figure(image_data, image_size=300):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    if image_data.shape != (128,96):
        image_data = np.reshape(image_data,(128,96))
    updated_img = px.imshow(image_data, height=image_size, binary_string=True)
#     updated_img = px.imshow(image_data, height=350, binary_string=True)
    updated_img.update_layout(dragmode="drawrect") # allows Pixclick function to take rectangular drag input
    updated_img.update_traces(hoverinfo='none',hovertemplate=None)
#     fig = go.Figure(updated_img)
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)
    
#     return fig
    return updated_img


def render_3d(img_data):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)

    # take in all_data['px_data'] entry and convert into 3d-rendered figure
    data = np.squeeze(img_data)

    threeD = np.repeat(data[:,:,np.newaxis], data.shape[1], axis=2) # 64x128 -> 64x128x64
    threeD[:,:,-1] = 0; threeD[:,:,0] = 0 # fills in front and back facing borders
    flipped_data = np.flip(threeD,axis=0)
    verts, faces, normals, values = measure.marching_cubes(np.transpose(flipped_data))
#     print(threeD.shape)
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='red',
        opacity=0.8,
    )
    
    layout = go.Layout(
        margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis = dict(visible=True),
            yaxis = dict(visible=True),
            zaxis =dict(visible=True),
            camera=dict(
                eye=dict(x=1.6, y=1.1, z=1.0)
            )
        )
    )
    data = [mesh]
    plot_figure = go.Figure(data=data, layout=layout)
    
    return plot_figure

def create_extrudedSTL(mode,img_data):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    # take in all_data['px_data'] entry and convert into .STL file
    data = np.squeeze(img_data)
    threeD = np.repeat(data[:,:,np.newaxis], data.shape[1], axis=2) # 64x128 -> 64x128x64
    threeD[:,:,-1] = 0; threeD[:,:,0] = 0 # fills in front and back facing borders
    flipped_data = np.flip(threeD,axis=0)
    verts, faces, normals, values = measure.marching_cubes(np.transpose(flipped_data))
    print(threeD.shape)
    obj_3d = mesh.Mesh(np.zeros(faces.shape[0],dtype=mesh.Mesh.dtype)) # Convert the faces into a mesh
    for i, f in enumerate(faces): # and then convert mesh to stl
        obj_3d.vectors[i] = verts[f]
    obj_3d.save(f'{mode}_final_choice.stl')

def revolve_3d(input_array):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    array2d = np.squeeze(input_array)
    array2d = array2d.transpose()
    # Check if the input is a 2D array
    if len(array2d.shape) != 2:
        raise ValueError("Input should be a 2D array")
    

    # Get the dimensions of the 2D array
    rows, cols = array2d.shape

    # Initialize a 3D array of zeros with shape (rows, cols, cols)
    array3d = np.zeros((rows, cols, cols))

    # Calculate the center row index
    center_row = rows // 2

    # Loop through each row and column of the 2D array
    for i in range(rows):
        for j in range(cols):
            # Calculate the value at the (i, j) position in 2D array
            value = array2d[i, j]

            # Loop through a circle in the 3D array and set the value
            for theta in range(360):
                angle = np.radians(theta)
                z = int(center_row + (i - center_row) * np.cos(angle))
                y = int(center_row + (i - center_row) * np.sin(angle))

                # Ensure the indices are in bounds
                if 0 <= z < rows and 0 <= y < cols:
                    array3d[z, j, y] = value
        array3d[:,-1,:] = 0 #forcing the bottom to be closed
        
    verts, faces, normals, values = measure.marching_cubes(np.transpose(array3d))
    print(array3d.shape)
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='red',
        opacity=0.7,
    )
    
    layout = go.Layout(
        margin=dict(t=0, b=0, l=0, r=0),
        scene=dict(
            xaxis = dict(visible=True),
            yaxis = dict(visible=True),
            zaxis =dict(visible=True),
            camera=dict(
                eye=dict(x=1.6, y=1.1, z=1.0)
            )
        )
    )
    data = [mesh]
    plot_figure = go.Figure(data=data, layout=layout)
    return plot_figure

def create_revolvedSTL(mode,img_data):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    array2d = np.squeeze(img_data)
    array2d = array2d.transpose()
    # Check if the input is a 2D array
    if len(array2d.shape) != 2:
        raise ValueError("Input should be a 2D array")
    
    rows, cols = array2d.shape
    array3d = np.zeros((rows, cols, cols))
    center_row = rows // 2

    for i in range(rows):
        for j in range(cols):
            # Calculate the value at the (i, j) position in 2D array
            value = array2d[i, j]
            # Loop through a circle in the 3D array and set the value
            for theta in range(360):
                angle = np.radians(theta)
                z = int(center_row + (i - center_row) * np.cos(angle))
                y = int(center_row + (i - center_row) * np.sin(angle))
                # Ensure the indices are in bounds
                if 0 <= z < rows and 0 <= y < cols:
                    array3d[z, j, y] = value
        array3d[:,-1,:] = 0 #forcing the bottom to be closed
        
    verts, faces, normals, values = measure.marching_cubes(np.transpose(array3d))
    # save as stl using vertices and faces
    obj_3d = mesh.Mesh(np.zeros(faces.shape[0],dtype=mesh.Mesh.dtype)) # Convert the faces into a mesh
    for i, f in enumerate(faces): # and then convert mesh to stl
        obj_3d.vectors[i] = verts[f]
    obj_3d.save(f'{mode}_final_choice.stl')



    
def compute_dimensions(image_data):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    # all calculations are done pixel-wise, then scaled up
        # all calculations are done pixel-wise, then scaled up
   
    if len(image_data.shape) != 2:
        image_data = np.squeeze(image_data)

    bw_threshold = 127.5
    plot_check = False # make this True if you want to plot images to see dimensions marked on them
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]

    # find seat height
    for row_idx in range(y_dim):
        row = image_data[row_idx,:]
        if np.any(row < bw_threshold):
            break
    height = y_dim - row_idx

    # find seat width
    for col_idx_sw in range(x_dim):
        col = image_data[row_idx - 5:row_idx +5,col_idx_sw]
        if np.any(col < bw_threshold):
            break
    seat_width = x_dim - (2*col_idx_sw)

    # find leg width
    for col_idx_lw in range(x_dim):
        col = image_data[y_dim-20: y_dim,col_idx_lw]
        if np.any(col < bw_threshold):
            break
    leg_width = x_dim - (2*col_idx_lw)

    if plot_check:

        print(f'Height {height}, Seat Width {seat_width}, Leg Width {leg_width}')
        image_data[row_idx,:] = 127.5
        image_data[:,col_idx_sw] = 127.5
        image_data[:,col_idx_lw] = 127.5
        plt.imshow(image_data, cmap='gray')
        plt.show()
       

    ratio = height/max(seat_width,leg_width)
    
    scale = 4 # 1px = 4mm
    return [height*scale,seat_width*scale,leg_width*scale,ratio]

def transform_to_cost(co2_value):
    # these statistics are from co2 of freeform 1k dataset
    # which are used to normalize existing designs + new designs
    min_co2 = 65.081834496576
    max_co2 = 140.63467500964796
    min_cost = 30
    max_cost = 150
    cost = ((co2_value - min_co2) / (max_co2 - min_co2)) * (max_cost - min_cost) + min_cost
    if cost < min_cost:
        cost = min_cost
    if cost > max_cost:
        cost = max_cost
        
    cost = np.round(cost)
        
    return cost

def frameworkCalcs(img_data,**kwargs):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    # This function provides a very rudimentary "slicing" algorithm for 3D printing.\
    # It's purpose is solely to calculate build time and mass, not to actually create gcode to be sent to the printer
    
    # Check if input(img_data) is a 2D numpy array
    if isinstance(img_data, np.ndarray) and len(img_data.shape) == 2:
        pass
    else:
        print("Input is not a 2D numpy array")
        return None
    
    # Check for specific keyword arguments and assign default values if they are not present
    numSolidLayers = kwargs.get('numSolidLayers',2)
    infillPercent = kwargs.get('infillPercent',0.1)
    nozzleSize = kwargs.get('nozzleSize',1.75)
    layerHeight = kwargs.get('layerHeight',1.0)
    roadWidth = kwargs.get('roadWidth',2.1)
    printSpeed = kwargs.get('printSpeed',30)
    materialDensity = kwargs.get('materialDensity',1.38/1000)
    outlineLayers = kwargs.get('outlineLayers',1)
    massFudgeFactor = kwargs.get('massFudgeFactor',1.22)
    timeFudgeFactor = kwargs.get('timeFudgeFactor',1.41)
    scale = kwargs.get('scale',4)
    
    # Scale data from 1 px = 1mm to 1 px = 4mm
    img_data = np.kron(img_data, np.ones((scale,scale)))
    
    #Estimate the perimiter (outline) length
    img = img_data.astype(np.uint8)
    #find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    perimeter = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        perimeter = perimeter + peri
    
    #Time & Mass Calculations
    crossSectionalArea = math.ceil(np.sum(img_data)) #Assume that each cell in the topology output is 1 unit of measure (e.g. millimeter), so the area is just summing all of the 1's. White = 1, black = 0
    solidLayerTime = (crossSectionalArea / (roadWidth * printSpeed))
    infillLayerTime = infillPercent * solidLayerTime
    numberInfillLayers = np.ceil((img_data.shape[1] / layerHeight)-numSolidLayers) #this assumes we're "extruding" the designs in a 3rd dimension (seat depth) of equal size to the seat width
    buildTime = (numSolidLayers-1) * solidLayerTime + 1*solidLayerTime * 1.4 + numberInfillLayers * infillLayerTime + ((perimeter/printSpeed)*(numSolidLayers-1+numberInfillLayers)*outlineLayers)+(perimeter/printSpeed)*1*outlineLayers
    buildTime = buildTime*timeFudgeFactor
    solidLayerMass = crossSectionalArea * layerHeight * materialDensity
    infillLayerMass  = infillPercent * solidLayerMass
    outlineMass = ((perimeter * layerHeight * roadWidth)*(numSolidLayers+numberInfillLayers)*outlineLayers)* materialDensity
    partMass = numSolidLayers * solidLayerMass + numberInfillLayers * infillLayerMass + outlineMass
    partMass = partMass*massFudgeFactor
    
    #LCA Calculations
    # https://docs.google.com/spreadsheets/d/1CR7ZsNzVr3kAr9luWDV-ccCX-kBhJtd-U8mbjh0Jjzs/edit?usp=sharing
    CO2saving = (partMass/1000)*30.3
#     equivMiles = CO2saving/0.35
    #equivCost = CO2saving*1.5 - 65
    equivCost = transform_to_cost(CO2saving)
    
    return [round(buildTime/3600,2), round(partMass/1000,4), CO2saving, round(equivCost,2)]
    
    
# -----------------------------------------------------------------------------------------
# PIXCLICK FUNCTIONS

   
# CHANGE_RECT FUNCTIONS (in order they're called)

def adjust_lvs(noise, latent_range, num_variations):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    noise = np.array(noise)
    if noise.shape[0] != 1:
        noise = noise.reshape(1,12)
    noise_variations = np.repeat(noise, num_variations, axis=0)
    for vector in noise_variations:
        for i in range(len(vector)):
            latent_change = np.random.randint(latent_range*10)/10 # latent change from 0 to 2
            direction = np.random.choice([-1,1])
            vector[i] += (direction * latent_change)
            while vector[i] > 1 or vector[i] < -1:
                vector[i] -= (direction * latent_change)
                latent_change = np.random.randint((latent_range/2)*10)/10 # more limited latent change # suggest edit
                direction = np.random.choice([-1,1])
                vector[i] += (direction * latent_change)
#     print("noise_variations dim:" + str(noise_variations.shape))
    return noise_variations

def adjust_imgs(noise_variations, current_bcs, num_variations, gen_model):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
#     print("current_bcs dim:" + str(current_bcs.shape))
    feature_bcs = np.tile(current_bcs, (num_variations,1))
#     print("noise_variations dim:" + str(noise_variations.shape))
#     print("feature_bcs dim:" + str(feature_bcs.shape))
    imgs = gen_model.predict([noise_variations,feature_bcs])
    imgs = 0.5 * imgs + 0.5
    return imgs

def find_black_counts(orig_img, imgs, selected_area, num_variations):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    top, bottom, left, right = selected_area
    rect_area = (bottom - top) * (right - left)
#     print(orig_img.shape)
#     img1 = orig_img[0]
    img_area = (np.shape(orig_img)[0] * np.shape(orig_img)[1]) - rect_area
    all_black_pixel_counts = []
    area_black_pixel_counts = []
    for i in range(num_variations):
        img2 = imgs[i]
        diff = cv2.absdiff(orig_img, img2)
     
        segmentation = diff >= 0.5
        area_black_count = np.count_nonzero(segmentation[top:bottom, left:right] != 0)
        area_ratio = (area_black_count / rect_area)*10
        area_black_pixel_counts.append(area_ratio)
        all_black_count = np.count_nonzero(segmentation != 0) - area_black_count
        all_ratio = (all_black_count / img_area)*10
        all_black_pixel_counts.append(all_ratio)
    return all_black_pixel_counts, area_black_pixel_counts

def feature_dist(img, current_bcs, imgs, disc_model, num_variations):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    current_bcs_reshape = current_bcs.reshape(1,12)
    feature_bcs = np.repeat(current_bcs_reshape, num_variations, axis=0)
    
    img_reshape = img.reshape(1,128,96,1)
    
#     print("img shape: " + str(img.shape))
#     print("img_reshape shape: " + str(img_reshape.shape))
#     print("bcs shape: " + str(current_bcs_reshape.shape))
#     print("imgs shape: " + str(imgs.shape))
#     print("feature_bcs shape: " + str(feature_bcs.shape))
    
    orig_prediction = disc_model.predict([img_reshape,current_bcs_reshape])
    predictions = disc_model.predict([imgs,feature_bcs])
    distances = []
    for i in range(len(imgs)):
        dist = euclidean(orig_prediction[0], predictions[i])
        distances.append(dist)
    return distances

def get_lv_idx(all_ratios, area_ratios, dist_scores):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    final_scores = []
    for i in range(len(area_ratios)):
        if area_ratios[i] > 0:
            score = (area_ratios[i]*3) - all_ratios[i] - dist_scores[i]
        else:
            score = -1000
        final_scores.append(score)
    final_scores_sorted = copy.deepcopy(final_scores)
    final_scores_sorted.sort(reverse=True)
    max_score= max(final_scores_sorted)
    if max_score > -1000:
        lv_idx = final_scores.index(max_score)
    else:
        lv_idx = False
    return lv_idx, final_scores

def gen_interp_imgs(imgs, current_lvs, current_bcs, noise_variations, lv_idx, num_interp, latent_stretch, gen_model):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    current_bcs_reshape = current_bcs.reshape(1,12)
    interpolated_bcs = np.repeat(current_bcs_reshape, num_interp, axis=0)
    
    current_lvs = np.array(current_lvs)
    current_lvs.resize((1,12),refcheck=False)
    #selected_img = imgs[lv_idx]
    p1 = current_lvs[0]
    p2 = noise_variations[lv_idx]
#     print(p1)
#     print(p2)
    interpolated_noise = interp_points(p1,p2,latent_stretch, num_interp)
#     print("noise shape: " + str(interpolated_noise.shape))
#     print("bcs shape: " + str(current_bcs_reshape.shape))
    interpolated_imgs = gen_model.predict([interpolated_noise,interpolated_bcs])
    interpolated_imgs = 0.5 * interpolated_imgs + 0.5
   
    
    interpolated_imgs = np.insert(interpolated_imgs, 0, interpolated_imgs[0], axis=0)
    interpolated_noise = np.insert(interpolated_noise, 0, interpolated_noise[0], axis=0)
    interpolated_bcs = np.insert(interpolated_bcs, 0, interpolated_bcs[0], axis=0)
    
    return interpolated_imgs, interpolated_noise
        
def interp_points(p1, p2,latent_stretch, n_steps):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    l1 = latent_stretch[0]
    l2 = latent_stretch[1]
    # interpolate ratios between the points
    ratios = np.linspace(l1, l2, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        # if you dont want to extend past lv range -1,1, uncomment below
        # but the outputs seem fine for some reason if -2 to 2?
        """for i in range(len(v)):
          if v[i] > 1:
            v[i] = 0.99999999
          elif v[i] < -1:
            v[i] = -0.9999999"""
        vectors.append(v)
    return np.asarray(vectors)

def remove_rect(changing_img):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    fig = px.imshow(changing_img[0].squeeze(),  color_continuous_scale='gray_r')
    fig.update_layout(dragmode="drawrect")
    fig.update(layout_coloraxis_showscale=False)
    return fig

def export_usage_log(mode,usage_log):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
    df = pd.DataFrame(usage_log)
    
    headers = 'tab,time,f1,f2'
    df.to_csv(f'{mode}_usage_log_{timestamp}.csv', index=False, header=headers)
    
def export_saved_img_data(mode,saved_imgs,saved_imgs_source,all_data):
    function_name = inspect.currentframe().f_code.co_name
    print_function(function_name)
    
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M")
    data = []
    for i in range(len(saved_imgs)):
        index = saved_imgs[i]
        index_data = np.hstack((all_data['lv_data'][index], all_data['cond_inputs'][index]))
        index_data = np.append(index_data, saved_imgs_source[i])
        data.append(index_data)
    data_array = np.vstack(data)
    
    df = pd.DataFrame(data_array)
    df.to_csv(f'{mode}_imgs_data_{timestamp}.csv', index=False)
    
    
    
    
    
