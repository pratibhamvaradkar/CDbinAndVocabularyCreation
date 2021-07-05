
#include <iostream>
#include <vector>

#include "torch/script.h" // One-stop header
#include "torch/torch.h"

// DBoW2
#include "DBoW2.h"
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h" // defines macros CVXX
#include "DVision/DVision.h"

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include "opencv2/nonfree/nonfree.hpp"

using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace cv;

void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height){

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));

    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
    
    descriptors.create(select_indice.size(), 32, CV_8U);

    for (int i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j<32; j++)
        {
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}


void loadFeatures(vector<vector<cv::Mat > > &features, cv::Mat descriptors);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void createVocabularyFile(OrbVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat > > &features);

// ----------------------------------------------------------------------------

int main()
{


    torch::DeviceType device_type;
    // device_type = torch::kCUDA;
    device_type = torch::kCPU;
    torch::Device device(device_type);

    // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("sm.pt");
    //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("/home/t/Dropbox/gcn_tiny.pt");
    torch::jit::script::Module module = torch::jit::load("../sm.pt");

    vector<cv::String> fn;
    glob("/home/pratibha/Downloads/TUM/rgbd_dataset_freiburg1_xyz/rgb/*.png", fn, false);

    size_t count = fn.size(); 

    vector<vector<cv::Mat > > features;
   
    int index = 0;
    for (size_t i=0; i<count; i++) //Looping through images
    {
        index++;
        
        if (index>0)
        {
            
            // std::cout << fn[i] << std::endl;
            cv::Mat image_now = cv::imread(fn[i],CV_LOAD_IMAGE_UNCHANGED);

            // cv::Mat img1;
            // image_now.convertTo(img1, CV_32FC1, 1.f / 255.f , 0); ///dont know why you've commented this/
            
            //Dimensions specific to TUM Dataset images above.
            int img_width = image_now.cols;
            int img_height = image_now.rows;
            // cout << "width = " << img_width << " height = " << img_height << endl;

            // Here we need to add following:
            //      take image as input from blob(?)
            //      calculate keypoints for this image
            //      convert the keypoints in appropriate format to be given to the model as input
            //      store the keypoints in pts
            //      store the descriptors in desc

            // //MSER blog detector
            // Ptr<cv::MSER> ms = cv::MSER::create();
            // std::vector<std::vector<cv::Point> > kypts;
            // std::vector<cv::Rect> mser_bbox;
            // ms->detectRegions(image_now, kypts, mser_bbox);

            // //plot blob regions for MSER
            // for (int i = 0; i < kypts.size(); i++)
            // {
            //     rectangle(image_now, mser_bbox[i], CV_RGB(0, 255, 0));  
            // }
            // imshow("mser", image_now);
            // waitKey(0);
            // return 0;

            // // Just to check what kypts (originally "regions") are exactly
            // for(auto row: kypts){
            //     // for(auto pt: row){
            //     //     // std::cout<<pt << " , ";
            //     // }
            //     std::cout << row.size()<< std::endl;
            //     std::cout << std::endl;
            // }
            // std::cout << "size : " <<kypts.size()<< "\n";
            // std::cout << "size : " <<kypts[0].size() << "\n";


            // //Extract SIFT keypoints
            // Ptr<SIFT> detector = SIFT::create();
            // std::vector<KeyPoint> kypts;
            // detector->detect(image_now, kypts);

            //Extract ORB keypoints
            const int MAX_FEATURES = 500;
            Ptr<Feature2D> detector = ORB::create();
            std::vector<KeyPoint> kypts;
            detector->detect(image_now, kypts);

//to be replaced: from here....
            
            // auto img1_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img1.data, {1, img_height, img_width, 1});
            // img1_tensor = img1_tensor.permute({0,3,1,2});
            // auto img1_var = torch::autograd::make_variable(img1_tensor, false).to(device);

            // std::vector<torch::jit::IValue> inputs;
            // inputs.push_back(img1_var);
            // auto output = module->forward(inputs).toTuple();

            // auto pts  = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
            // auto desc = output->elements()[1].toTensor().to(torch::kCPU).squeeze();

//  to here....

            //for every keypoint get the corresponding patch
            std::vector<torch::jit::IValue> inputs;

            int n_kp = kypts.size();

            for(size_t j=0; j<n_kp; j++)
            {
                //xx and yy must be the keypoint values
                int xx = kypts[j].pt.x ;
                int yy = kypts[j].pt.y ;

                if (xx >= 32 && yy >= 32 && xx <= (img_width-32) && yy <= (img_height-32))
                {
                    //Extracting the patch
                    // cv::Rect rect = cv::Rect(xx-32, yy-32, xx+32, yy+32);
                    cv::Rect rect = cv::Rect(xx-32, yy-32, 64, 64);
                    cv::Mat patch = image_now(rect);
                    //Converting the patch to a Tensor and then into IValue which is used to feed the network
                    // at::Tensor tensor_patch = torch::from_blob(patch.data, { patch.rows, patch.cols, 1}, at::kByte).unsqueeze_(0).unsqueeze_(0);
                    at::Tensor tensor_patch = torch::from_blob(patch.data, {1, patch.rows, patch.cols, 1}, at::kByte);
                    tensor_patch = tensor_patch.to(at::kFloat);
                    inputs.push_back(tensor_patch);
                    // cout<<"input size = " << inputs.size() << endl;
                    // cout << "patch tensor size = " << tensor_patch.sizes() << endl; // tensor_patch.sizes() for shape and tensor_patch.dtype() for data type of individual values

                }
            }

            // auto pts  = kypts.toTensor().to(torch::kCPU).squeeze();
            // auto desc = module.forward(inputs).toTensor().to(torch::kCPU).squeeze();
            // cout << typeid(module.forward(inputs)).name();


            // // cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data<float>());
            // // // cv::Mat pts_mat(cv::Size(3, mser_pts.size(0)), CV_32FC1, mser_pts.data<float>());
            // // cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data<unsigned char>());

            // // int border = 8;
            // // int dist_thresh = 4;

            // // std::vector<cv::KeyPoint> keypoints;
            // // cv::Mat descriptors;
            // // nms(pts_mat, desc_mat, keypoints, descriptors, border, dist_thresh, img_width, img_height);
            // // loadFeatures(features, descriptors);
            
        }
    }

    cout << "... Extraction done!" << endl;

    // define vocabulary
    const int nLevels = 6;
    const int k = 10; // branching factor
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    OrbVocabulary voc(k, nLevels, weight, score);

    std::string vocName = "vocCDbin.bin";
    createVocabularyFile(voc, vocName, features);

    cout << "--- THE END ---" << endl;

    return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, cv::Mat descriptors)
{


    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void createVocabularyFile(OrbVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat > > &features)
{

  cout << "> Creating vocabulary. May take some time ..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "> Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "> Saving vocabulary..." << endl;
  // voc.saveToBinaryFile(fileName);
  cout << "... saved to file: " << fileName << endl;
}
// ----------------------------------------------------------------------------
