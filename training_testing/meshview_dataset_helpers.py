
import numpy as np
import scipy
import skimage.transform
import cv2
import scipy.io as sio
import csv

def load_imCoord(visFile):
    print(visFile)
    matcontents = sio.loadmat(visFile)
    #visFC = matcontents['visibleFC']
    #visFC = visFC.astype(int)
    imCoord = matcontents['imCoord']
    return imCoord       # array rows  = meshIDS, columns are x, y position in images

def load_imCoord_alt(visFile):
        print(visFile)
        matcontents = sio.loadmat(visFile)
        #visFC = matcontents['visibleFC']
        #visFC = visFC.astype(int)
        imCoord_x = matcontents['imCoord_x']
        imCoord_y = matcontents['imCoord_y']
        return (imCoord_x, imCoord_y)       # array rows  = meshIDS, columns are x, y position in images

def load_anns(annsFile):
    meshID_anns = {}
    with open(annsFile,'r') as f:
        reader=csv.reader(f,delimiter='\t')
        for row in reader:
            meshID = int(row[0]) - 1 # CONVERT from matlab ones based indexing to zero based indexing
            meshID_anns[meshID] = int(row[1])

    return meshID_anns  #dictionary: keys meshIDs, values annotation

def load_imgfiles(infile, base_dir):
    f = open(infile,'r')
    base_files = f.read().splitlines()
    f.close()

    imgFileNames = []
    for img in base_files:
        fullpath = base_dir + img[2:]
        imgFileNames.append(fullpath)
    return imgFileNames  #list of image file names


def create_pt_datastruct(imCoord):
    '''create data structure pts which holds locations where each mesh face was seen in an image
    pts = [meshFaceID imgID x y]
    does this by manipulating the sparse matrices visFC (indicator matrix where a 1 in the row, col indicates the meshFaceID (row) was seen in the image (col)
    and imCoord which is a nFaces x 2*nImgs sparse matrix
    where the x position where Face 'i' is seen in image 'j' can be found at row = i, col = 2*j
    and   the y position where Face 'i' is seen in image 'j' can be found at row = i, col = 2*j+1 (zero based indexing)
    '''
    #x_old = imCoord[:,::2]
    #y_old = imCoord[:,1::2]
    dims = imCoord.shape
    nRow = dims[0]
    nCol = int(dims[1]/2)
    print("nCol %d" %nCol)
    x = scipy.sparse.csc_matrix((nRow, nCol), dtype=np.float32)
    y = scipy.sparse.csc_matrix((nRow, nCol), dtype=np.float32)
    print(dims)
    for i in range(0, nCol):  #this is slow, but using straightforward indexing (x = imCoord[:,::2]) appears to require a cryptic dense intermediate matrix which causes memory to be used up for large matrices
        x[:,i] = imCoord[:,2*i]
        y[:,i] = imCoord[:,1+(2*i)]

    a = scipy.sparse.find(x)
    b = scipy.sparse.find(y)

    #assert np.array_equal(a[0], b[0])  #double check indexing is consistent - testing showed it was
    #assert np.array_equal(a[1], b[1])

    pts = np.stack((a[0], a[1], a[2], b[2]))
    pts = np.transpose(pts)

    #small dataset for testing
    #pts = pts[0:1500,:]

    return pts
def create_pt_datastruct_alt(imCoord_x, imCoord_y):
    '''create data structure pts which holds locations where each mesh face was seen in an image
    pts = [meshFaceID imgID x y]
    does this by manipulating the sparse matrices visFC (indicator matrix where a 1 in the row, col indicates the meshFaceID (row) was seen in the image (col)
    and imCoord which is a nFaces x 2*nImgs sparse matrix
    where the x position where Face 'i' is seen in image 'j' can be found at row = i, col = 2*j
    and   the y position where Face 'i' is seen in image 'j' can be found at row = i, col = 2*j+1 (zero based indexing)
    '''
    a = scipy.sparse.find(imCoord_x)
    b = scipy.sparse.find(imCoord_y)

    #assert np.array_equal(a[0], b[0])  #double check indexing is consistent - testing showed it was
    #assert np.array_equal(a[1], b[1])

    pts = np.stack((a[0], a[1], a[2], b[2]))
    pts = np.transpose(pts)

    #small dataset for testing
    #pts = pts[0:1500,:]

    return pts

def group_pts_by_meshID(pts, STRUCT):
    ''' take pts data struct [meshFaceID imgID x y] and reorganize into ptsByMeshID dictionary'
    keys are meshID ints, value for each key is a list of lists where
    the parent list contains all views of the mesh,
    each indivdual list holds info about each view [imgID x y]'''
    print("grouping points by meshID")
    ptsByMeshID = dict()
    for pt in pts:
        meshID = pt[0].astype(int)

        if meshID in ptsByMeshID:
            ptsByMeshID[meshID].append([pt[1].astype(int), pt[2], pt[3]]) #value of each key (meshID) is list of lists
        else:
            ptsByMeshID[meshID] = [[pt[1].astype(int), pt[2], pt[3]]] #list of lists each element of primary list contains imgID, x, y point, defining a 'view' of a given mesh element

    #restructure dictionary as a list of lists
    if STRUCT == 'List':
       meshElmViews = []
       for elm in ptsByMeshID:
         meshElmViews.append([elm, ptsByMeshID[elm]])

   # print("finished grouping points by meshID")
    if STRUCT == 'List':
       return meshElmViews

    if STRUCT == 'Dict':
       return ptsByMeshID


def extract_patch(im, patch_radius, x , y):  #extracts a square patch 2*patch_radius at point pti from image im
    ''' extracts a square patch (patch_radius x patch_radius pixels) surrounding point pti in image im
    rescales pixel intensities by dividing by pixScale.
    if the patch interescts edges it is rescaled to be a square (using skimage.transform.resize)
    '''
    height, width, channels = im.shape
    x_lb = int(max(0, x - patch_radius))
    x_ub = int(min(width-1, x + patch_radius))
    y_lb = int(max(0, y - patch_radius))
    y_ub = int(min(height-1, y + patch_radius))

    patch = im[y_lb:y_ub, x_lb:x_ub]

    if (patch.shape[0] != patch_radius*2 or patch.shape[1] != patch_radius*2):
        patch = cv2.resize(patch, (patch_radius*2, patch_radius*2)) #resize patch

    return patch

def select_annotated_views(n, viewDict, meshID_anns, offset):
    meshElmViews = []
    missing = 0  #number of annotated views not found in the mesh (shouldn't happen, but occasionally does)
    for elm in meshID_anns:
        ann = meshID_anns[elm]

        if elm in viewDict:  #elm should be in viewDict; but occasionally there are problems with back projection
            views = viewDict[elm]  #modify image number in views to account for offset
            for i in range(len(views)):
                views[i][0] = views[i][0] + offset
            meshElmViews.append([n, elm, ann, views])
        else:
            missing = missing + 1

    print("could not find %d annotations in mesh (this number should be a very small fraction of total)" % missing)
    return meshElmViews

def create_train_segment(n, imCoord,meshID_anns, offset):
    pts   =  create_pt_datastruct(imCoord)
    views =  group_pts_by_meshID(pts,'Dict')
    train_segment = select_annotated_views(n,views, meshID_anns, offset)
    return train_segment

def create_train_segment_alt(n, imCoord_x, imCoord_y,meshID_anns, offset):
    pts   =  create_pt_datastruct_alt(imCoord_x, imCoord_y)
    views =  group_pts_by_meshID(pts,'Dict')
    train_segment = select_annotated_views(n,views, meshID_anns, offset)
    return train_segment
