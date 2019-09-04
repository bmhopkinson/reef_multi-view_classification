function [Fcenters, visibleFC, imCoord_x, imCoord_y ] = facesVisibletoCameras_nanoRT(Cam, pCamCalib, V, F,fileBase)

addpath(genpath('/home/cv-bhlab/Documents/MATLAB/Library/3D_Reconstruction/mesh_utils'));  %read agisoft camera files, projects points etc
%determine which points on a mesh are visible in cameras based on
%projection into frame and clear line of sight

%addpath(genpath('/home/brian/Documents/MATLAB/Library/geom3d'));
%addpath('../Common');
PHOTO_PATH_PREFIX = './data/photos/';
PHOTO_EXT = '.png';
%% Load Cameras %%

nCamAln = size(Cam,2);

%export image list file
fn_imagelist = strcat(fileBase,'image_list.txt');
imlist = fopen(fn_imagelist,'w');
for i = 1: nCamAln
    img_path = strcat(PHOTO_PATH_PREFIX, Cam(i).label, PHOTO_EXT);
    fprintf(imlist,'%s\n', img_path);
end
fclose(imlist);

% calculate face centers %%

nFaces = size(F,1);
Fcenters = zeros(nFaces, 3);
for i = 1:nFaces
    pt1 = V(F(i,1),:); pt2 = V(F(i,2),:); pt3 = V(F(i,3),:);
    Fcenters(i,:) = (pt1+pt2+pt3)/3; %centroid of triangle
end


%% create aabb tree on faces
bl = V(F(:,1),:); %lower bound of aabb box - initialize to values of 1st vertex
bu = V(F(:,1),:); %upper bound of aabb box - initialize to values of 1st vertex

for i = 2:size(F,2)
    bl = min(bl, V(F(:,i),:));
    bu = max(bu, V(F(:,i),:));
end

tr = maketree([bl, bu]);  % from D. Engwirda's aabb tree matlab library
%% use aabb tree to determine faces relevant to each camera
% and then project those face centers using full camera moError using containers.
tic;
[visByCam, imCoordByCam_x, imCoordByCam_y] = projectFacesToCams(Cam,pCamCalib, V,F, tr);

fprintf(1,'finished aabb tree testing\n');
toc
% save('checkpoint.mat');
%load('checkpoint.mat');
%% check for line of sight using nanoRT based bounding-volume hierarchy
tic

parfor j = 1:nCamAln
    if isempty(visByCam{j})
        continue;
    end

    Fsub = F(visByCam{j},:);
    %remap vertices so only relevant vertices need to be passed
    Vidx = Fsub(:);
    Vsub = V(Vidx,:);

    refInds = zeros(size(Vidx));
    for k = 1:size(Vidx,1)
        refInds(Vidx(k))= k;
    end
    Fsub = refInds(Fsub);

    if(size(visByCam{j},1) > 1) %in rare cases only one face is visible - it can't be blocked and passing a single face in messes up nanort_los_test
      to_elim_cpp = nanort_los_test(single(Vsub),uint32(Fsub), single(Cam(j).camPos));
      to_elim_cpp = double(to_elim_cpp) + 1 ; % change from zero based indexing to ones based and convert data type

      j_temp = visByCam{j};
      x_temp = imCoordByCam_x{j};
      y_temp = imCoordByCam_y{j};

      j_temp(to_elim_cpp) = [];
      x_temp(to_elim_cpp) = [];
      y_temp(to_elim_cpp) = [];

    else    %only one face is visible - accept it
      j_temp = visByCam{j};
      x_temp = imCoordByCam_x{j};
      y_temp = imCoordByCam_y{j};
    end


    visByCam{j} = j_temp;
    imCoordByCam_x{j} = x_temp;
    imCoordByCam_y{j} = y_temp;

end  % end loop on cameras
fprintf(1,'time to test for line of sight using nanort\n');
toc


%% %%%% GENERATE SPARSE MATRICES

%calculate total views
total_views = 0;
for j = 1:nCamAln
    total_views = total_views + length(visByCam{j});
end

idx_vis = 0;    %keep track of position in indices vectors
i_vis = zeros(total_views,1);  %row indicies
j_vis = zeros(total_views,1);  %col indicies
x_data = zeros(total_views,1);
y_data = zeros(total_views,1);


for j = 1:nCamAln
    %assemble visibleFC data and imCoord data. imCoord_x and _y are
    % nFaces x nCams matrices holding respectively x and  y coordinate of row_face in col_camera.
    nelms = length(visByCam{j});
    vs = idx_vis + 1;
    ve = idx_vis + nelms;
    i_vis(vs:ve) = visByCam{j};
    j_vis(vs:ve) = j*ones(nelms,1);
    x_data(vs:ve) = imCoordByCam_x{j};
    y_data(vs:ve) = imCoordByCam_y{j};

    idx_vis = idx_vis + nelms;  %update index

end

%strip off unused zeros from preallocated index vectors
i_vis = i_vis(1:idx_vis);
j_vis = j_vis(1:idx_vis);
x_data = x_data(1:idx_vis);
y_data = y_data(1:idx_vis);

visibleFC = sparse(i_vis,j_vis,ones(idx_vis,1));
imCoord_x = sparse(i_vis,j_vis, x_data);
imCoord_y = sparse(i_vis,j_vis, y_data);

totVis = sum(visibleFC,2); %total number of images in which a point is visible;
obscuredPts = find(totVis == 0);
C = zeros(nFaces, 3);
C(obscuredPts,2) = 255; %green
%
figure
pcshow(Fcenters, C);

end
