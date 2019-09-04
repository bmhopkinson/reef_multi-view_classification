// line of sight test between camera center and mesh faces
// a mex file for use with matlab
// makes use of nanoRT bounding volume heirarchy for interesction tests

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "nanort_mod.h"
#include "nanort_los_test.h"
#include "MatlabDataArray.hpp"
#include<vector>

using namespace std;
class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
      checkArguments(outputs, inputs);

      matlab::data::Array V = inputs[0];
      matlab::data::Array F = inputs[1];
      matlab::data::Array camPos = inputs[2];
      matlab::data::ArrayDimensions dimV = V.getDimensions();
      matlab::data::ArrayDimensions dimF = F.getDimensions();
      int nV = dimV[0];  //number of vertices
      int nF = dimF[0];  //number of faces

      // transfer vertices to flat array (for nanoRT)
      float* vertices = new float[3*nV];
      for(int i=0; i<nV; i++){
        vertices[3*i + 0] = V[i][0];
        vertices[3*i + 1] = V[i][1];
        vertices[3*i + 2] = V[i][2];
      }

      unsigned int* faces = new unsigned int[3*nF];
      for(int i=0; i<nF; i++){
        faces[3*i + 0] = static_cast<unsigned int>(F[i][0]) - 1;  //convert from matlab ones based indexing to c++ zero-based indexing
        faces[3*i + 1] = static_cast<unsigned int>(F[i][1]) - 1;
        faces[3*i + 2] = static_cast<unsigned int>(F[i][2]) - 1;
      }


      nanort::BVHBuildOptions<float> build_options; // Use default option
        //build_options.cache_bbox = false;
      nanort::TriangleMesh<float>    triangle_mesh(vertices, faces, sizeof(float) * 3);
      nanort::TriangleSAHPred<float> triangle_pred(vertices, faces, sizeof(float) * 3);
      nanort::BVHAccel<float, nanort::TriangleMesh<float>, nanort::TriangleSAHPred<float>, nanort::TriangleIntersector<> > accel;

      bool ret = false;
      ret = accel.Build(nF, build_options, triangle_mesh, triangle_pred);
      assert(ret);

      nanort::BVHBuildStatistics stats = accel.GetStatistics();
/*
      std::cout << "  BVH statistics:\n";
      std::cout << "    # of leaf   nodes: " <<  stats.num_leaf_nodes << "\n";
      std::cout << "    # of branch nodes: " <<  stats.num_branch_nodes << "\n";
      std::cout << "  Max tree depth     :"  <<  stats.max_tree_depth << "\n";
      float bmin[3], bmax[3];
      accel.BoundingBox(bmin, bmax);
      std::cout << "  Bmin               : " << bmin[0] <<" "<< bmin[1] << " " << bmin[2] << "\n";
      std::cout << "  Bmax               : " << bmax[0] <<" "<< bmax[1]<< " " << bmax[2] << "\n";

*/
      vector<unsigned int> notseen;
      for(int j = 0; j < nF; j++){  //loop over faces
          nanort::Ray<float> ray;

          ray.org[0] = camPos[0];
          ray.org[1] = camPos[1];
          ray.org[2] = camPos[2];
  //        cout << "camPos x: " << ray.org[0] << " " << ray.org[1] << " " << ray.org[2] << "\n";
          float3 org;
          org[0] = camPos[0];
          org[1] = camPos[1];
          org[2] = camPos[2];

          //destination of ray is at the center of face of interest, first calculate face center (centroid) then assign
          float3 dest;
          int v1 = faces[3*j+0];  int v2 = faces[3*j+1];  int v3 = faces[3*j+2];

          float xc = (vertices[3*v1+0] + vertices[3*v2+0] + vertices[3*v3+0])/3.0;
          float yc = (vertices[3*v1+1] + vertices[3*v2+1] + vertices[3*v3+1])/3.0;
          float zc = (vertices[3*v1+2] + vertices[3*v2+2] + vertices[3*v3+2])/3.0;
      //    cout << "fsub: " << j << " face center : " << xc << " " << yc << " " << zc << "\n";
          dest[0] = xc;
          dest[1] = yc;
          dest[2] = zc;

          float3 dir = dest - org;
          dir.normalize();
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          nanort::TriangleIntersector<> triangle_intersector(vertices, faces, sizeof(float) * 3);
          nanort::BVHTraceOptions trace_options;
          bool hit = accel.Traverse(ray, trace_options, triangle_intersector); //traverse finds closest hit along ray (i.e. direct line of sight - MultiHitTraverse will give all the interesctions

          if(hit){
              unsigned int fid = triangle_intersector.intersection.prim_id;
              unsigned int thisf = static_cast<unsigned int>(j);
              if(fid != thisf){  //does first face hit (fid) equal current face - if not there is no line of sight for current face-camera pair
                notseen.push_back(thisf);
              }
          }

      } //end loop on faces

//  std::cout << "finished checking for lines of sight\n";

  delete [] vertices;  //memory allocated for mesh
  delete [] faces;

  matlab::data::ArrayFactory factory;  // following example at the bottom of this page: https://www.mathworks.com/help/matlab/apiref/matlab.data.arrayfactory.html
  auto buff = factory.createBuffer<uint32_t>(notseen.size());
  uint32_t* buffPtr = buff.get();
  for_each(notseen.begin(), notseen.end(), [&](const uint32_t& e) {*(buffPtr++) = e;} );
  matlab::data::TypedArray<uint32_t> notseen_ml= factory.createArrayFromBuffer({notseen.size(), 1}, std::move(buff));

  outputs[0] = notseen_ml;

}

    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        matlab::data::ArrayFactory factory;

        if (inputs.size() != 3) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Two inputs required") }));
        }

        if (inputs[0].getType() != matlab::data::ArrayType::SINGLE ||
            inputs[0].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input face matrix must be type single") }));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::UINT32) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input face matrix must be type uint32") }));
        }

        if (inputs[2].getType() != matlab::data::ArrayType::SINGLE) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Input face matrix must be type single") }));
        }

        if (inputs[0].getDimensions().size() != 2 ||
            inputs[1].getDimensions().size() != 2 ||
            inputs[2].getDimensions().size() != 2 ) {
            matlabPtr->feval(matlab::engine::convertUTF8StringToUTF16String("error"),
                0, std::vector<matlab::data::Array>({ factory.createScalar("Inputs do not have correct dimensions") }));
        }

  }
};
