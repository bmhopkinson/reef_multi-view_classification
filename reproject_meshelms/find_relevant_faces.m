function [Frel, node_rel] = find_relevant_faces(T, calib, tr)
%find faces that potentially project into the camera (T - camera transfor, calib - camera calibration)
% also return relavant node ids in node_rel
%accelerated with the aid aabb tree (tr);
 BUFF = 0.1; %fraction of width and height to allow as a buffer for considering points to be sufficiently close to examine further
 
 fx = calib.fx;
 fy = calib.fy;
 cx = calib.cx;
 cy = calib.cy;
 w  = calib.width;
 h  = calib.height;

 %subset faces into those relevant for current camera
 queue.data = [1];  %start by enqueing root node 
 queue.head = 1; 
 queue.tail = 2;
 Frel = []; %relevant faces
 node_rel = []; %relevant node ids
 
 while(queue.head < queue.tail)
   i = queue.data(queue.head);
   queue.head =queue.head+1;

   if(tr.ii(i,2) == 0)   %leaf node (no children); append faces to consider later
      Frel = [Frel ; tr.ll{i}];
      node_rel = [node_rel, i];
   else %not a leaf node - see if children should be considered
      c1 = tr.ii(i,2);  %first child node of binary tree;
      c2 = c1 + 1;      %second child node is always 1 below 
      cnodes = [c1 c2];

      for k = 1:size(cnodes,2)

        %project corners of aabb box into camera  - NEED TO PROJECT ALL 8 CORNERS OF BOX 
        tmp = reshape(tr.xx(cnodes(k),:), 3,2);

        %permute to get all 8 corners of aabb box
        crn = zeros(3,8);
        nc = 1;
        for aa = 1:2  %must be a better way to do this
          for bb = 1:2
            for cc = 1:2
                crn(:,nc) = [tmp(1,aa); tmp(2,bb);tmp(3,cc)];
                nc = nc + 1;
             end
          end
        end
        crn(4,:) = ones(1,8);  %convert to homogeneous coordinates

        pCam = T*crn;  %convert world points to local camera coordinates
        pinh = pCam(1:2,:)./pCam(3,:);  %scale by z-distance from camera 

        % use pinhole projections for basic sanity check (can get some points way outside the camera view
        % projecting into camera with nonlinear corrections (radial distortion etc)
        x_pinhole = cx + pinh(1,:)*fx;
        y_pinhole = cy + pinh(2,:)*fy;

        if(sum(x_pinhole < (w*(1+BUFF))) == 0 || sum(x_pinhole > (-w*BUFF)) == 0)
          continue; 
        elseif(sum(y_pinhole < (h*(1+BUFF))) == 0 || sum(y_pinhole > (-w*BUFF)) == 0)
          continue;
        else
          %pass - a portion of aabb box projects into camera
          %enque this node
          queue.data(queue.tail) = cnodes(k);
          queue.tail = queue.tail +1;
        end
      end %and for loop on child nodes

   
   end %end if on leaf vs. child node

 end %end while

end

