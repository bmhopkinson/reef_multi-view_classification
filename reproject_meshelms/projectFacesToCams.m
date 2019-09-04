function [visByCam, imCoordByCam_x, imCoordByCam_y] = projectFacesToCams(Cam,pCamCalib, V,F, tr, varargin)
% use aabb tree to determine faces relevant to each camera
% and then project those face centers using full camera model.

dist_threshold = [];
%process varargin
params_to_variables = containers.Map(...
    {'Dist_Threshold'},...
    {'dist_threshold'});

v = 1;
while v <= numel(varargin)
  param_name = varargin{v};
  if isKey(params_to_variables,param_name)
    assert(v+1<=numel(varargin));
    v = v+1;
    % Trick: use feval on anonymous function to use assignin to this workspace 
    feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
  else
    error('Unsupported parameter: %s',varargin{v});
  end
  v=v+1;
end

%first calculate face centers
nFaces = size(F,1);
Fcenters = zeros(nFaces, 3);
for i = 1:nFaces
    pt1 = V(F(i,1),:); pt2 = V(F(i,2),:); pt3 = V(F(i,3),:);
    Fcenters(i,:) = (pt1+pt2+pt3)/3; %centroid of triangle
end

visByCam = {};
imCoordByCam_x = {};
imCoordByCam_y = {};

if ~isempty(dist_threshold)
    distByCam = {};
end


nCams = size(Cam,2);

parfor j = 1:nCams
    
    [Frel,~] = find_relevant_faces(Cam(j).Tinv, pCamCalib(Cam(j).sensor_id), tr);
    
    Fcsub = Fcenters(Frel,:);
    nFcsub = size(Fcsub,1);
    
    nv_cam = 0;
    j_vis_cam = zeros(nFcsub,1);
    x_cam = zeros(nFcsub,1);
    y_cam = zeros(nFcsub,1);
    
    w = pCamCalib(Cam(j).sensor_id).width;
    h = pCamCalib(Cam(j).sensor_id).height;
    
    for i = 1:nFcsub
        [x,y, x_pinhole, y_pinhole] = projectPointToCamera(Fcsub(i,:), Cam(j).Tinv, pCamCalib(Cam(j).sensor_id));
        if(x_pinhole > -0.3*w && x_pinhole < 1.3*w && y_pinhole > -0.3*h && y_pinhole < 1.3*h) %use pinhole projection as sanity check, nonlinear corrections can erroneously project locations way outside of field of view into the image
            if(x > 0 && x < w && y > 0 && y < h)
              nv_cam = nv_cam+1; %increment total number of views
              % add values to index vectors for visibleFC matrix
              j_vis_cam(nv_cam) = Frel(i);
              x_cam(nv_cam) = x;
              y_cam(nv_cam) = y;
                            
            end
        end %end pinhole sanity check
    end  %end loop on faces
    
    %trim trailing zeros
    j_vis_cam = j_vis_cam(1:nv_cam);
    x_cam     = x_cam(1:nv_cam);
    y_cam     = y_cam(1:nv_cam);
    
    if ~isempty(dist_threshold)   %threshold by distance between face and camera
        Fc = Fcenters(j_vis_cam,:);
        camPos = Cam(j).camPos';
        
        sqdist = sum((Fc - repmat(camPos,size(Fc,1),1)).^2,2);
        dist = sqrt(sqdist);
        dist_sort = sort(dist,'ascend');
        n_avg = ceil( 0.1 *size(dist_sort,1)); %average closest 10% 
        dist_near = dist_threshold * mean(dist_sort(1:n_avg));
        
        pass = (dist < dist_near);
        j_vis_cam(~pass) = [];
        x_cam(~pass)     = [];
        y_cam(~pass)     = [];
        
    end    
        
    visByCam{j} = j_vis_cam;
    imCoordByCam_x{j} = x_cam;
    imCoordByCam_y{j} = y_cam;
    
end %end loop on cameras 


end

