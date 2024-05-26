#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#define PI 3.1415926

/*本函数对原始3Dbox可视化*/
using namespace std;
using namespace pcl;
using namespace cv;

int delay = 1;
struct CloudResult
{
    std::string name;
    float xc;
    float yc;
    float zc;
    float l;
    float w;
    float h;
    float yaw;
    float cls_id;
	float score;
};
std::vector<CloudResult> Inference_result;

std::string data_dir = "../dataset_x_dali/";
// const std::vector<std::string> class_name = {"Car", "Sightseeing", "patrol_cars", "Truck", "Bus", "Van", "Pedestrian",
//                                              "Cyclist", "Tricyclist", "Motorcyclist", "Barrowlist", "Trafficcone", "ignore"};
const std::vector<std::string> class_name = {"person", "pushing", "bike", "rider", "car", "truck", "bus", "Special"};

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Lidar Viewer"));
pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

long int flag = 0;
void LabelParsing(std::vector<std::string> &object_vector);
void Visualize_pointcloud_results();

void SplitString(const std::string &s, std::vector<std::string> &v, const std::string &c)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void ReadLabelTxt(std::string label_path, std::vector<std::string> &label_vector)
{
    ifstream infile;
    std::string temp;
    infile.open(label_path.data());
    while (getline(infile, temp))
    {
        label_vector.push_back(temp);
    }
    infile.close();
}

void LidarParsing(int tempnum)
{
    for (int i = 0; i < tempnum; i++)
    {
        pcl::PointXYZRGBA PointTemp1;
        PointTemp1.x = input_cloud->points[i].x;
        PointTemp1.y = input_cloud->points[i].y;
        PointTemp1.z = input_cloud->points[i].z;
		PointTemp1.a = 255;
		int intensity = static_cast<int>(input_cloud->points[i].intensity);

        //根据点云强度PointTemp1.a转换为rgb显示
        // if (intensity <= 63)
        // {
        //     PointTemp1.r = 0;
        //     PointTemp1.g = 254 - 4 * PointTemp1.a;
        //     PointTemp1.b = 255;
        // }
        // else if (intensity > 63 && intensity <= 127)
        // {
        //     PointTemp1.r = 0;
        //     PointTemp1.g = 4 * PointTemp1.a - 254;
        //     PointTemp1.b = 510 - 4 * PointTemp1.a;
        // }
        // else if (intensity > 127 && intensity <= 191)
        // {
        //     PointTemp1.r = 4 * PointTemp1.a - 510;
        //     PointTemp1.g = 255;
        //     PointTemp1.b = 0;
        // }
        // else if (intensity > 191 && intensity <= 255)
        // {
        //     PointTemp1.r = 255;
        //     PointTemp1.g = 1022 - 4 * PointTemp1.a;
        //     PointTemp1.b = 0;
        // }
        PointTemp1.r = 255;
        PointTemp1.g = 255;
        PointTemp1.b = 255;
        cloud->points.push_back(PointTemp1);
    }
}

void ReadFileList(std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    std::string pcd_path = data_dir + "pcd/";
    if ((dp=opendir(pcd_path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }
    
    while ((dirp=readdir(dp)) != NULL)
    {
        if(strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
        {
            continue;
        } 
        files.push_back(dirp->d_name);
    }

    closedir(dp);
    sort(files.begin(),files.end());
}


int main()
{
	viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGBA> (cloud,"sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 50, 0, 0, 0);
	
    std::vector<std::string> filenames;
    ReadFileList(filenames);

    size_t num=0;
    for (size_t i = num; i < filenames.size(); i++)
    {
        std::string cloud_file = data_dir + "pcd/" + filenames[i];
        std::string tmp_file = cloud_file;
		std::string img_tmp_file = cloud_file;

        // std::vector<std::string> char_arr;
        // SplitString(filenames[i], char_arr, ".");
        
        std::string label_file = tmp_file.replace(tmp_file.find("pcd"), 3, "label/txt");
        label_file = label_file.replace(label_file.find("pcd"), 3, "txt");

        //std::string img_file = img_tmp_file.replace(img_tmp_file.find("pcd"), 3, "img");
        //img_file = img_file.replace(img_file.find("pcd"), 3, "jpg");

        //VideoCapture capture(img_file);
        //cv::Mat frame;
        //capture >> frame;
        //imshow("display img", frame);
        //cv::waitKey(delay);

        std::vector<std::string> label_vector;
        ReadLabelTxt(label_file, label_vector);
        std::cout << label_file.c_str() << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_file, *input_cloud) == -1)
        {
            printf("Couldn't read %s\n", cloud_file.c_str());
            return (-1);
        }
        int tempnum = input_cloud->points.size();
        LidarParsing(tempnum);
        LabelParsing(label_vector);
        Visualize_pointcloud_results();		

		Inference_result.clear();
        input_cloud->points.clear();
        cloud->points.clear();
        if (viewer->wasStopped())
        {
        	return 0;
        }
        printf("num=%ld\n",num);
        num++;
    }
    return 0;
}


void LabelParsing(std::vector<std::string> &object_vector)
{
	for (size_t i = 0; i < object_vector.size(); i++)
    {
		CloudResult temp_obj;
        std::vector<std::string> char_arr;
        SplitString(object_vector[i], char_arr, " ");
        temp_obj.name = char_arr[0];
        temp_obj.xc = atof(char_arr[1].c_str());
        temp_obj.yc = atof(char_arr[2].c_str());
        temp_obj.zc = atof(char_arr[3].c_str());
        temp_obj.l  = atof(char_arr[4].c_str());
        temp_obj.w  = atof(char_arr[5].c_str());
        temp_obj.h  = atof(char_arr[6].c_str());
        temp_obj.yaw = atof(char_arr[7].c_str());
        // temp_obj.cls_id = atof(char_arr[7].c_str());
		// temp_obj.score = atof(char_arr[8].c_str());
		Inference_result.push_back(temp_obj);
    }
}


void Visualize_pointcloud_results()
{
    for (int i = 0; i < (int)Inference_result.size(); i++)
    {
        //std::string cls_name = class_name[int(Inference_result[i].cls_id)];
        std::string cls_name = Inference_result[i].name;
        float centerX = Inference_result[i].xc;
        float centerY = Inference_result[i].yc;
        float centerZ = Inference_result[i].zc;
        float length  = Inference_result[i].l;
        float width   = Inference_result[i].w;
        float height  = Inference_result[i].h;
        float heading = Inference_result[i].yaw;
		// float score = Inference_result[i].score;
        
        Eigen::Quaternionf rotation(cos(heading / 2), 0, 0, sin(heading / 2));
        Eigen::Vector3f location(centerX, centerY, centerZ);

		float r, g, b;
        // if (cls_name == "Car")
        // {
        //     r = 1.0;
        //     g = 0.0;
        //     b = 1.0;
        // }
        // else if(cls_name == "Sightseeing")
        // {
        //     r = 1.0;
        //     g = 0.56;
        //     b = 0.12;
        // }
        // else if(cls_name == "patrol_cars")
        // {
        //     r = 0.78;
        //     g = 0.0;
        //     b = 0.56;
        // }
        // else if (cls_name == "Truck")
        // {
        //     r = 0.0;
        //     g = 1.0;
        //     b = 0.0;
        // }
        // else if (cls_name == "Van")
        // {
        //     r = 0.0;
        //     g = 0.0;
        //     b = 1.0;
        // }
        // else if (cls_name == "Bus")
        // {
        //     r = 1.0;
        //     g = 1.0;
        //     b = 0.0;
        // }
        // else if (cls_name == "Pedestrian")
        // {
        //     r = 0.0;
        //     g = 1.0;
        //     b = 1.0;
        // }
        // else if (cls_name == "Cyclist")
        // {
        //     r = 1.0;
        //     g = 0.0;
        //     b = 0.0;
        // }
        // else if (cls_name == "Tricyclist")
        // {
        //     r = 1.0;
        //     g = 0.38;
        //     b = 0.0;
        // }
        // else if (cls_name == "Motorcyclist")
        // {
        //     r = 0.85;
        //     g = 0.43;
        //     b = 0.84;
        // }
        // else if (cls_name == "Barrowlist")
        // {
        //     r = 0.0;
        //     g = 0.78;
        //     b = 0.55;
        // }
        // else if (cls_name == "Trafficcone")
        // {
        //     r = 0.12;
        //     g = 0.56;
        //     b = 1.0;
        // }
        // else if (cls_name == "ignore")
        // {
        //     r = 1.0;
        //     g = 1.0;
        //     b = 1.0;     
        // }
        
        if (cls_name == "person")
        {
            r = 1.0;
            g = 0.0;
            b = 1.0;
        }
        else if(cls_name == "pushing")
        {
            r = 1.0;
            g = 0.56;
            b = 0.12;
        }
        else if(cls_name == "bike")
        {
            r = 0.30;
            g = 0.0;
            b = 0.66;
        }
        else if (cls_name == "rider")
        {
            r = 0.0;
            g = 1.0;
            b = 0.0;
        }
        else if (cls_name == "car")
        {
            r = 0.0;
            g = 0.0;
            b = 1.0;
        }
        else if (cls_name == "truck")
        {
            r = 1.0;
            g = 1.0;
            b = 0.0;
        }
        else if (cls_name == "bus")
        {
            r = 0.0;
            g = 1.0;
            b = 1.0;
        }
        else if (cls_name == "sign")
		{
			r = 0.12;
            g = 0.56;
            b = 1.0;
        }
        else if (cls_name == "Special")
        {
            r = 1.0;
            g = 0.0;
            b = 0.0;
        }
       
        char strCube[100] = { 0 };
		sprintf(strCube, "cube%ld", flag++);
        viewer->addCube(location, rotation, length, width, height, strCube, 0);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, strCube);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, strCube);
        viewer->setRepresentationToWireframeForAllActors();
		// viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, strCube);
		// viewer->setRepresentationToSurfaceForAllActors(); 

        pcl::PointXYZ pos_dist = pcl::PointXYZ(location(0) + length / 2, location(1), location(2) + height / 2);
		char strObj[100] = { 0 };
		double dDist = sqrt(location(0) * location(0) + location(1) * location(1));
        float angle = atan2f(location(0), location(1)) * 180 / PI;	//方位角
        if((centerX < 0 && centerY > 0) || (centerX < 0 && centerY < 0))
        {
            angle = angle + 360;
        }
        sprintf(strObj, "%s", cls_name.c_str());
        // sprintf(strObj, "%0.1f", score);	
        // sprintf(strObj, "%0.1f", location(1));
        sprintf(strObj, "%s_%0.1f", cls_name.c_str(),location(1));
		char strText[100] = { 0 };
		sprintf(strText, "dist%ld_", flag++);
		viewer->addText3D(strObj, pos_dist, 0.5, r, g, b, strText);

		// 画方向箭头
		// std::vector<pcl::PointXYZ> points;
		// points.emplace_back(pcl::PointXYZ(length/2, width/2, height/2));
		// points.emplace_back(pcl::PointXYZ(length/2, -width/2, -height/2));

		// pcl::PointXYZ pt1(location(0), location(1), location(2));
		// Eigen::Vector3f pointTemp((points.at(0).x + points.at(1).x) / 2.0f,
        //                         (points.at(0).y + points.at(1).y) / 2.0f,
        //                         (points.at(0).z + points.at(1).z) / 2.0f);
        // Eigen::Matrix3f matrix;
		// matrix = Eigen::AngleAxisf(heading, Eigen::Vector3f::UnitZ());	
		// Eigen::Vector3f center = matrix * pointTemp + location;
		// pcl::PointXYZ pt2(center(0),center(1),center(2));

		// char strYaw[100] = { 0 };
		// sprintf(strYaw, "yaw%ld_", flag++);
		// viewer->addArrow<pcl::PointXYZ>(pt2, pt1, r, g, b, false, strYaw, 0);
    }
    
    viewer->updatePointCloud(cloud);
    viewer->addPointCloud(cloud);
    viewer->spinOnce(delay);
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
}
