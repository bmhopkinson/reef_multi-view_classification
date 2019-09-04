// project 3D points into camera defined by projective camera model
// mex file for use with matlab
// CURRENTLY THIS IS SLOWER!!!! THAN SIMPLY DOING IT IN MATLAB - perhaps due to all the copying of variables could try converting to refs, but probably it's not worth it. 

#include "mex.hpp"
#include "mexAdapter.hpp"
#include<cmath>
#include<vector>
#include "MatlabDataArray.hpp"

using namespace std;
using namespace matlab::data;

// move struct declaration and function declaration to header file
struct ProjectedPt
{
  double xpin;
  double ypin;
  double x;
  double y;
};



class MexFunction : public matlab::mex::Function {
public:
    ProjectedPt projectPointToCamera(double ptWorld[3], matlab::data::TypedArray<double> Tcam, matlab::data::StructArray camCalib){
   // apply projective camera model with radial and tangential distortion
     ProjectedPt pt;

     //unpack calibration data for clarity
     
     Array fx_ml = camCalib[0]["fx"]; 
     Array fy_ml = camCalib[0]["fy"];
     Array cx_ml = camCalib[0]["cx"];
     Array cy_ml = camCalib[0]["cy"];
     Array k1_ml = camCalib[0]["k1"];
     Array k2_ml = camCalib[0]["k2"];
     Array k3_ml = camCalib[0]["k3"];
     Array p1_ml = camCalib[0]["p1"];
     Array p2_ml = camCalib[0]["p2"];
     
     double fx = fx_ml[0];
     double fy = fy_ml[0];
     double cx = cx_ml[0];
     double cy = cy_ml[0];
     double k1 = k1_ml[0];
     double k2 = k2_ml[0];
     double k3 = k3_ml[0];
     double p1 = p1_ml[0];
     double p2 = p2_ml[0];
     
     
     
     // convert world points to local camera coordinates - matrix multiply pt3D by Tcam
     double ptCam[3];
     ptCam[0] = Tcam[0][0]*ptWorld[0] + Tcam[0][1]*ptWorld[1] + Tcam[0][2]*ptWorld[2] + Tcam[0][3];  
     ptCam[1] = Tcam[1][0]*ptWorld[0] + Tcam[1][1]*ptWorld[1] + Tcam[1][2]*ptWorld[2] + Tcam[1][3];  
     ptCam[2] = Tcam[2][0]*ptWorld[0] + Tcam[2][1]*ptWorld[1] + Tcam[2][2]*ptWorld[2] + Tcam[2][3];  
     //ptCam[3] = Tcam[3][0]*pt3D[0] + Tcam[3][1]*pt3D[1] + Tcam[3][2]*pt3D[2] + Tcam[3][3];  // homogeneous "1.0" no need to actually compute
     
     // scale by z-distance from camera
     double ptNrm[2];
     ptNrm[0] = ptCam[0]/ptCam[2];  
     ptNrm[1] = ptCam[1]/ptCam[2]; 
     
     double r = pow((pow(ptNrm[0],2) + pow(ptNrm[1],2)), 0.5);  // radial distance of pt from camera center

     // use pinhole projections for basic sanity check (can get some points way outside the camera view
     // projecting into camera with nonlinear corrections (radial distortion etc)
     pt.xpin = cx + ptNrm[0]*fx;
     pt.ypin = cy + ptNrm[1]*fy;

     // apply non-linear distortion corrections
     double r_cor = 1.000 + k1*pow(r, 2) + k2*pow(r,4) + k3*pow(r,6); 
     double xp = ptNrm[0]*r_cor + (p1*(pow(r,2) + 2*pow(ptNrm[0],2)) + 2*p2*ptNrm[0]*ptNrm[1]);
     double yp = ptNrm[1]*r_cor + (p2*(pow(r,2) + 2*pow(ptNrm[1],2)) + 2*p1*ptNrm[0]*ptNrm[1]);
     
     pt.x = cx + xp*fx;
     pt.y = cy + yp*fy;

     return pt;
   }
    
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
      checkArguments(outputs, inputs);
     // matlab::data::Array points3D = inputs[0];
     // matlab::data::Array Tcam = inputs[1];
      matlab::data::TypedArray<double> points3D = inputs[0];
      matlab::data::TypedArray<double> Tcam = inputs[1];
      matlab::data::StructArray camCalib = inputs[2];

      matlab::data::ArrayDimensions dim_points3D = points3D.getDimensions();
      int nPoints = dim_points3D[0];
      
      vector<int> vis_cam;
      vector<double> x_cam;
      vector<double> y_cam;
      
      Array w_ml = camCalib[0]["width"];
      Array h_ml = camCalib[0]["height"];
      double w = w_ml[0];
      double h = h_ml[0];

      //double w = camCalib[0]["width"];
      //double h = camCalib[0]["height"];
     // ProjectedPt pt = projectPointToCamera(pt1, Tcam, camCalib);
     // cout << "projected pt x, y: " << pt.x << " " << pt.y << "\n";
     // cout << "projected pt xpin, ypin " << pt.xpin << " " << pt.ypin << "\n";
             
      int nv_cam = 0; // total number of cameras in view
      for(int i = 0; i<nPoints; i++){
          double x = points3D[i][0];
          double y = points3D[i][1];
          double z = points3D[i][2];
          double pt1[3] = {x, y, z};
          
          ProjectedPt pt = projectPointToCamera(pt1, Tcam, camCalib);
          if(pt.xpin > -0.3*w && pt.xpin < 1.3*w && pt.ypin > -0.3*h && pt.ypin < 1.3*h){ //use pinhole projection as sanity check, nonlinear corrections can erroneously project locations way outside of field of view into the image
            if(pt.x > 0 && pt.x < w && pt.y > 0 && pt.y < h){
               nv_cam++; //increment total number of views
               // add values to index vectors for visibleFC matrix
               vis_cam.push_back(i);
               x_cam.push_back(pt.x);
               y_cam.push_back(pt.y);
            }
        } //end pinhole sanity check
      }  //end loop on faces
    
    //repackage data for output to matlab
    matlab::data::ArrayFactory factory;  // following example at the bottom of this page: https://www.mathworks.com/help/matlab/apiref/matlab.data.arrayfactory.html
    auto buff1 = factory.createBuffer<uint32_t>(vis_cam.size());
    uint32_t* buff1Ptr = buff1.get();
    for_each(vis_cam.begin(), vis_cam.end(), [&](const uint32_t& e) {*(buff1Ptr++) = e;} );
    matlab::data::TypedArray<uint32_t> vis_cam_ml= factory.createArrayFromBuffer({vis_cam.size(), 1}, std::move(buff1));
    
    auto buff2 = factory.createBuffer<double>(x_cam.size());
    double* buff2Ptr = buff2.get();
    for_each(x_cam.begin(), x_cam.end(), [&](const double& e) {*(buff2Ptr++) = e;} );
    matlab::data::TypedArray<double> x_cam_ml= factory.createArrayFromBuffer({x_cam.size(), 1}, std::move(buff2));
    
    auto buff3 = factory.createBuffer<double>(y_cam.size());
    double* buff3Ptr = buff3.get();
    for_each(y_cam.begin(), y_cam.end(), [&](const double& e) {*(buff3Ptr++) = e;} );
    matlab::data::TypedArray<double> y_cam_ml= factory.createArrayFromBuffer({y_cam.size(), 1}, std::move(buff3));
    
    outputs[0] = vis_cam_ml;
    outputs[1] = x_cam_ml;
    outputs[2] = y_cam_ml;
    
   }// end () operator calculate

   

   void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
       std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
       matlab::data::ArrayFactory factory;

       if (inputs.size() != 3) {
           matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
               0, std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required") }));
       }

       if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE  ) {
           matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
               0, std::vector<matlab::data::Array>({ factory.createScalar("Input set of pts must be matrix of doubles") }));
       }

       if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE ) {
           matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
               0, std::vector<matlab::data::Array>({ factory.createScalar("Input transform must be of type double") }));
       }

       if (inputs[2].getType() != matlab::data::ArrayType::STRUCT) {
           matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
               0, std::vector<matlab::data::Array>({ factory.createScalar("3rd argument must be pCamCalib struct array") }));
       }

       if (inputs[0].getDimensions().size() != 2 ||
           inputs[1].getDimensions().size() != 2  ) {
           matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
               0, std::vector<matlab::data::Array>({ factory.createScalar("Inputs do not have correct dimensions") }));
       }

    }  // end checkArguments
}; //end class delaraction
