function [Vsets_out, Fsets_out] = split_cameras_mesh(Cam, pCamCalib, V,F,depth, fileBase)


%group cameras spatially (e.g. aabb tree) then split mesh into exclusive
%groups of faces associated with camera groups
% 
% overall approach is:
% 1. group cameras spatially using aabb tree - at two levels: major groups,
% minor groups
% 2. idenfity faces viewed by each group (minor)
% 3. determine overlap of faces between minor groups
% 4. exclusively allocate faces to minor groups using greedy algorithm-
% when faces are acquired from a group, add that group's cameras to
% acquiring group
% 5. Aggregate minor groups to major

% tunable parameters
ALLOC_METRIC = 0;  %metric to use for allocating faces to camera groups: 0 = min distance; 1 = avg distance; 2 = weighting of views and avg distance
DIST_MULT = 1.5; %if distance between camera and face group exceeds DIST_MULT * median_dist, the view is excluded from consideration
CAM_ADD_THRESHOLD = 1; % threshold at which to add minor group cameras to major group owning a particular faceGroup
DIST_WEIGHTING = 10; %weighting to give distance relative to number of views for face group partitioning

%% 1. split cameras spatially used aabb tree into major and minor groups

n_groups = 2^depth;
nCams = size(Cam, 2);

%extract camera positions
camPos = zeros(nCams,3);

for i=1:nCams
    camPos(i,:) = Cam(i).camPos;
end


node_min = max(10, round(nCams/(n_groups*10)));
op.nobj = node_min;
tr = maketree([camPos, camPos],op);  % from D. Engwirda's aabb tree matlab library

%descend tree to split depth and then collect all leaf nodes contained by
%nodes at 'level'

parents = [1]; %start with root node

for i=1:depth
    children = [];
    for j = 1:size(parents,2)
        c1 = tr.ii(parents(j),2);  %first child
        c2 = c1 + 1;  %second child 
        children = [children, c1, c2];
    end
    parents = children;
    
end

major_groups = parents;

%define minor groups by descending say several levels further (need to
%consider encountering leaf nodes)
%collect all cameras of these minor groups
%retain minor-> major map

minor_depth = 4;
minor_groups = cell(n_groups,1);
minor_to_major.major = [];
minor_to_major.minor = [];
for i = 1:n_groups
  parents = major_groups(i);  %starting node - major_group id
  
  for j=1:minor_depth
    children = [];
    for k = 1:size(parents,2)
        c1 = tr.ii(parents(k),2);  %first child
        if c1 == 0  %leaf node
           children =[children, parents(k)]; %don't descend, reappend leaf node 
           continue;
        else
           c2 = c1 + 1;  %second child 
           children = [children, c1, c2]; %append new nodes
        end
    end
    parents = children;
    
  end
  minor_groups{i} = parents;
  minor_to_major.major = [minor_to_major.major i*ones(1,size(parents,2))];
  minor_to_major.minor = [minor_to_major.minor linspace(1,size(parents,2),size(parents,2))];
end

camGroups = {};      % major x minor cell array in which each cell holds array of camera indices corresponding to minor group
tic;
for i=1:n_groups
   parents = minor_groups{i};
   n_minor = size(parents,2);
   camGrp_minor = {};
   
   for j = 1:n_minor
    this_parent = parents(j);
    queue.data = this_parent;
    queue.head = 1;
    queue.tail = 2;
    
    camIDs = [];
    
    while(queue.head < queue.tail)
      nc = queue.data(queue.head); %current node
      queue.head =queue.head+1;
      
      if(tr.ii(nc,2) == 0)   %leaf node (no children); append cameras
        camIDs = [camIDs ; tr.ll{nc}];
      else %not a leaf node - append to queue for descent
        c1 = tr.ii(nc,2);  %first child node of binary tree;
        c2 = c1 + 1;      %second child node is always 1 below 
        cnodes = [c1 c2];
        queue.data = [queue.data, cnodes];
        queue.tail = queue.tail + 2;
      end
      
    end  %end while for queue processing
    camGroups{i,j} = camIDs;
   end %end loop on minor groups
   
end %end for on major groups
toc
fprintf(1,'done defining camera groups\n');

%% 2.calculate number of views and distances between camGroups and faceGroups
tic;
%create aabb tree on faces 
bl = V(F(:,1),:); %lower bound of aabb box - initialize to values of 1st vertex
bu = V(F(:,1),:); %upper bound of aabb box - initialize to values of 1st vertex

for i = 2:size(F,2)
    bl = min(bl, V(F(:,i),:));
    bu = max(bu, V(F(:,i),:));
end

op.nobj = 1000;  %max number of objects in leaves - avg number seems to be ~70% of this 
tr_faces = maketree([bl, bu],op);  % from D. Engwirda's aabb tree matlab library

Fsets = {}; 
visByCam_group = {};

% first do this on minor groups

n_minor = size(camGroups,2);
n_nodes = size(tr_faces.ii,1);  %total nodes, not just leaf, but should be ok. 
n_totgrps = n_groups*n_minor;
stats_views = zeros(n_totgrps,n_nodes);
stats_dists = cell(n_totgrps, n_nodes);
for i=1:n_groups

    for j = 1:n_minor
      if(isempty(camGroups{i,j})) 
          continue; 
      end
      idx_m = n_minor*(i-1) + j;
      %stats_views = containers.Map('KeyType','double','ValueType','double');
      %stats_dist  = containers.Map('KeyType','double','ValueType','double');
      
      camsMinor = Cam(camGroups{i,j});
      n_cm = size(camsMinor,2);
      for k = 1:n_cm
          [~,node_ids] = find_relevant_faces(camsMinor(k).Tinv, pCamCalib(camsMinor(k).sensor_id), tr_faces);
          for l = 1:size(node_ids,2)
              stats_views(idx_m, node_ids(l)) =  stats_views(idx_m, node_ids(l)) + 1; %increment views
              
              bb_cen = (tr_faces.xx(node_ids(l),1:3) + tr_faces.xx(node_ids(l),4:6)) ./2; %center of bounding box 
              dist_cambb = sum((camsMinor(k).camPos - bb_cen').^2).^0.5;
              stats_dists{idx_m, node_ids(l)} =   [stats_dists{idx_m, node_ids(l)} dist_cambb];
              
         end  %end loop on node_ids
              
      end  %end loop on cams in minor group
      
    end  %end loop on n_minor

end  %end loop on n_groups



% filter views - currently throw out views/dists more than DIST_MULT * median

for i = 1:n_nodes
    all_dists = [];
    nonempty = [];
    for j = 1:n_totgrps %loop through all cam groups to collect view distances
        cam_dists = stats_dists{j,i};
        if(isempty(cam_dists))
            continue;
        end
        all_dists =  [all_dists cam_dists];
        nonempty = [nonempty j];
    end
    
    if(isempty(all_dists)) % not a leaf node
        continue;
    end
    
    %determine median and threshold
    med = median(all_dists);    
    dthresh = DIST_MULT*med;
    
    nne = size(nonempty,2);
    
    for j =1:nne  %loop through cam groups and remove views exceeding dthresh
        cam_idx = nonempty(j);
        cam_dists = stats_dists{cam_idx,i};
        to_remove = cam_dists > dthresh;
        n_rem = sum(to_remove);
        
        if(n_rem ~= 0)
          stats_views(cam_idx,i) = stats_views(cam_idx,i) - n_rem;
          stats_dists{cam_idx,i} = cam_dists(~to_remove);
        end
        
    end   
end

toc
fprintf(1,'done assessing views between camGroups and faceGroups\n');

%% 3. allocate faces to camGroups

camGroup_min_nodes = cell(n_totgrps,1);
camGroup_min_faces = cell(n_totgrps,1);

for i = 1:n_nodes
    n_vbyg = stats_views(:,i);
    if(sum(n_vbyg) == 0)  %not a leaf node, skip
        continue;
    end
    
    dist_byg = zeros(n_totgrps,1);
    
    if ALLOC_METRIC == 0 %minimum distance
      for j = 1:n_totgrps
        this_dist = min(stats_dists{j,i});
        if(isempty(this_dist))
            continue;
        end
        
        dist_byg(j) = min(this_dist);
      end
   
      metric = 1./dist_byg;
      metric(metric == Inf) = 0;
    
    elseif ALLOC_METRIC == 1  %average distance
       for j = 1:n_totgrps
           dist_byg(j) = mean(stats_dists{j,i});
       end
       metric = 1./dist_byg;
       
    elseif ALLOC_METRIC == 2 %weight avg distance and number of views
       for j = 1:n_totgrps
           dist_byg(j) = mean(stats_dists{j,i});
       end
       
       metric = n_vbyg + DIST_WEIGHTING .* (1./dist_byg);
    end
    
    [~,idx_win] = max(metric);
    camGroup_min_nodes{idx_win} = [camGroup_min_nodes{idx_win} i];
    camGroup_min_faces{idx_win} = [camGroup_min_faces{idx_win}; tr_faces.ll{i}];
    
end

% agregate to level of major groups
camGroup_nodes = cell(n_groups,1);
camGroup_faces = cell(n_groups,1);
camGroups_mj = cell(n_groups,1);
for i = 1:n_groups
    temp_nodes = [];
    temp_faces = [];
    for j = 1:n_minor
        idx_m = n_minor*(i-1) + j;
        temp_nodes = [temp_nodes camGroup_min_nodes{idx_m}];
        temp_faces = [temp_faces; camGroup_min_faces{idx_m}];
        camGroups_mj{i} = [camGroups_mj{i}; camGroups{i,j}];
    end

    camGroup_nodes{i} = temp_nodes;
    camGroup_faces{i} = temp_faces;
end

%augment major camera groups with cameras from minor groups that view faces
%in major camera group
for i = 1: n_groups
    cams_add = [];
    nodes_ofg = camGroup_nodes{i};
    n_nofg = size(nodes_ofg,2);
    mask = ones(n_totgrps,1);
    s_idx = n_minor*(i-1) + 1;
    e_idx = n_minor*i;
    mask(s_idx:e_idx) = 0;
    
    for j = 1:n_nofg
        %minor_views= mask.*stats_views(:,nodes_ofg(j));  %# of views turns out to be a bad metric; remove views in major group using mask
        idx_min = [];
        min_dist = 1E10;  %very large distance
        for k = 1:n_totgrps
            cam_dists = stats_dists{k,nodes_ofg(j)};
            if(isempty(cam_dists) || mask(k) == 0) %skip if no views or if this camera is in the same major group
                continue;
            end
            
            cam_min = min(cam_dists);
            if(cam_min < min_dist)
                idx_min = k;
                min_dist = cam_min;
            end
            
            
        end
        
        idx_mj = ceil(idx_min/n_minor);
        idx_mi = mod( idx_min,n_minor);
        if (idx_mi == 0)
            idx_mi = n_minor;
        end
        
        cams_add = [cams_add; camGroups{idx_mj, idx_mi}];

    end
    
    camGroups_mj{i} = unique([camGroups_mj{i}; cams_add]);

end


% remap faces and vertices for split mesh sections
Fsets_out = {};
Vsets_out = {};
for i = 1:n_groups
    Fsub = F(camGroup_faces{i},:); 
    %remap vertex indices in faces
    Vidx = unique(Fsub(:));
    Vsub = V(Vidx,:);
    
    refInds = zeros(size(Vidx));
    for k = 1:length(Vidx)
        refInds(Vidx(k))= k;
    end
    
    Fsub = refInds(Fsub);
    CamSub = Cam(camGroups_mj{i}); 
    outfile = strcat(fileBase,'camGrp_',num2str(i),'.mat');
    save(outfile, 'CamSub','pCamCalib','Vsub','Fsub','-v7.3');
    
    Fsets_out{i} = Fsub;
    Vsets_out{i} = Vsub;
    
end


% 
%% plot results 
%  multiple plots
colors = {'r','g','b','c','m','y'};

for i = 1:n_groups
   figure(i);
   this_group = camGroups_mj{i};
   for j = 1:size(this_group,1)
       plotCircle3D(camPos(this_group(j),:),[0, 0, 1],0.1, colors{mod(i,size(colors,2))+1});
       hold on;
   end
   pcshow(Vsets_out{i},colors{mod(i,size(colors,2))+1});
   plot_box3d(tr.xx(major_groups(i),1:3), tr.xx(major_groups(i),4:6),colors{mod(i,size(colors,2))+1});
   
   face_nodes = camGroup_nodes{i};
   for j = 1:size(face_nodes,2)
       plot_box3d(tr_faces.xx(face_nodes(j),1:3), tr_faces.xx(face_nodes(j),4:6),colors{mod(i,size(colors,2))+1});
   end
   
   
   trisurf(F, V(:,1), V(:,2), V(:,3));
   hold off;
end

% single plot - faces only

figure;

trisurf(F, V(:,1), V(:,2), V(:,3),'FaceColor',[0,0,0]);
hold on;
for i = 1:n_groups
    pcshow(Vsets_out{i},colors{mod(i,size(colors,2))+1});
end




end

