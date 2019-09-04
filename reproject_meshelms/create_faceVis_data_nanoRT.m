%create data structures that map faces to camera images and the points where they appear in those images

addpath(genpath('~/Documents/MATLAB/Library/matGeom/matGeom'));
addpath('~/Documents/MATLAB/Library/xml');
addpath('~/Documents/MATLAB/3D_Reconstruction/mesh_utils');

% camFile  = './data/hog_reef/hog_reef_20190108_cameras.xml';
% camVersion = 'v1.4';
% meshFile = './data/hog_reef/hog_reef_20190108_mesh.off';
% fileBase = './data/hog_reef/hog_reef_20190108_';

% camFile  = './data/Crescent_simple_cameras.xml';
% camVersion = 'v1.4';
% meshFile = './data/Crescent_simple_mesh.off';
% fileBase = './data/Crescent_simple_';

% camFile  = './data/0441_simple_3_cameras.xml';
% camVersion = 'v1.4';
% meshFile = './data/0441_simple_3_mesh.off';
% fileBase = './data/0441_simple_3';


% camFile = './data/0443_02122016_cameras.xml';
% camVersion = 'v1.2';
% meshFile = './data/0443_02122016_mesh.off';
% fileBase = './data/0443_02122016_';
% 
camFile = './data/crescent_reef_refined_20190129/crescent_reef_refined_20190129_cameras.xml';
camVersion = 'v1.4';
meshFile = './data/crescent_reef_refined_20190129/crescent_reef_refined_20190129_mesh.off';
fileBase = './data/crescent_reef_refined_20190129/crescent_reef_refined_20190129_';

SPLIT_MESH = 1;

[V, F] = readMesh_off(meshFile);
[Cam, pCamCalib] = loadCameraData(camFile);

if SPLIT_MESH == 1
    depth = 2;
    [Vsets, Fsets] = split_cameras_mesh(Cam, pCamCalib, V,F,depth,fileBase);
else
    Vsets{1} = V;
    Fsets{1} = F;
end

n_groups = size(Vsets,2);

%write out split meshes
for i = 1:n_groups
    fn_meshout = strcat(fileBase,'mesh_',num2str(i),'.off');
    writeMesh_off(fn_meshout,Vsets{i},Fsets{i});
end

for i = 1:n_groups
    if SPLIT_MESH == 1
     grp_infile = strcat(fileBase,'camGrp_',num2str(i),'.mat');
     load(grp_infile); %loads  'CamSub','pCamCalib','Vsub','Fsub'
     Cam = CamSub;
    else 
        Vsub = V;
        Fsub = F;
    end
    
    fn_imglist = strcat(fileBase,num2str(i),'_');
    [Fcenters, visibleFC, imCoord_x, imCoord_y ] = facesVisibletoCameras_nanoRT(Cam, pCamCalib, Vsub, Fsub, fn_imglist);
    outfile = strcat(fileBase,'faceVis_',num2str(i),'.mat');

    save(outfile,'Cam','pCamCalib','Fcenters','visibleFC','imCoord_x','imCoord_y','-v7.3');
end


