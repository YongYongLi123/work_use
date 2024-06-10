#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>

#include <dlfcn.h>
#include <LidarTracking.h>
#include <boost/thread/thread.hpp>
#define pi 3.1415926

using namespace std;

struct CloudResult
{
    std::string name;
    float xc;
    float yc;
    float zc;
    float w;
    float l;
    float h;
    float yaw;
    float score;
};
std::vector<CloudResult> Inference_result;

std::string data_dir   = "../data/";
std::string video_dir  = "../data/yolov-5x-0.3.avi";
cv::VideoCapture capture(video_dir);
void Image_display(cv::Mat &show_image, float *img_uv, size_t &pixel_size);
void point2pixel(void **data, double *rotateArray, float *translateArray, double *cameraArray, unsigned int *length);

std::string index_path = "../data/trainval.txt";
std::vector<std::string> index_vector;
const std::vector<std::string> class_name =  {"car", "truck", "bus", "non_motor_vehicles", "pedestrians", "other_obstacles"};

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Lidar Viewer"));
pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

long int flag = 0;
void Parse_label(std::vector<std::string> &object_vector);

void (*LidarTracking)(std::vector<Cluster>, std::vector<Tracking>&, float);

void Visualize_pointcloud_results(std::vector<Tracking> &trackingList);

double cameraArray[9] = {1423.542622768, 0, 941.1427120394005,
                         0, 1424.134977058, 278.053427613, 
                         0, 0, 1};
float translateArray[3] = {0, -0.16, 0};
float rotateAngle[3] = {90, -1.7, 0};
double pitch   = rotateAngle[0] * CV_PI / 180.0;
double roll    = rotateAngle[1] * CV_PI / 180.0;
double heading = rotateAngle[2] * CV_PI / 180.0;
double rotateArray[9] = {0}; 


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

std::string Convert(float Num)
{
    std::ostringstream oss;
    oss << Num;
    std::string str(oss.str());
    return str;
}

void read_label_txt(std::string label_path, std::vector<std::string> &label_vector)
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

void read_index_txt()
{
    std::string temp;
    ifstream infile;
    infile.open(index_path.data());
    while (getline(infile, temp)) //获取一行内容,并且赋值给temp
    {
        cout << typeid(temp).name() << endl;
        index_vector.push_back(temp);
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
        PointTemp1.a = input_cloud->points[i].intensity;

        //根据点云强度PointTemp1.a转换为rgb显示
        if (PointTemp1.a <= 63)
        {
            PointTemp1.r = 0;
            PointTemp1.g = 254 - 4 * PointTemp1.a;
            PointTemp1.b = 255;
        }
        else if (PointTemp1.a > 63 && PointTemp1.a <= 127)
        {
            PointTemp1.r = 0;
            PointTemp1.g = 4 * PointTemp1.a - 254;
            PointTemp1.b = 510 - 4 * PointTemp1.a;
        }
        else if (PointTemp1.a > 127 && PointTemp1.a <= 191)
        {
            PointTemp1.r = 4 * PointTemp1.a - 510;
            PointTemp1.g = 255;
            PointTemp1.b = 0;
        }
        else if (PointTemp1.a > 191 && PointTemp1.a <= 255)
        {
            PointTemp1.r = 255;
            PointTemp1.g = 1022 - 4 * PointTemp1.a;
            PointTemp1.b = 0;
        }
        PointTemp1.a = 255;
        cloud->points.push_back(PointTemp1);
    }
}


int main()
{
    void *trackingHandle = dlopen("./LidarTracking.so", RTLD_LAZY);
	LidarTracking = (void(*)(std::vector<Cluster>, std::vector<Tracking>&, float))dlsym(trackingHandle, "LidarTracking");

	viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGBA> (cloud,"sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 50, 0, 0, 0);

	rotateArray[0] = cos(heading) * cos(roll);
    rotateArray[1] = sin(heading) * cos(roll);
    rotateArray[2] = -sin(roll);
    rotateArray[3] = -sin(heading) * cos(pitch) + cos(heading) * sin(roll) * sin(pitch);
    rotateArray[4] = cos(heading) * cos(pitch) + sin(heading) * sin(roll) * sin(pitch);
    rotateArray[5] = cos(roll) * sin(pitch);
    rotateArray[6] = sin(heading) * sin(pitch) + cos(heading) * sin(roll) * cos(pitch);
    rotateArray[7] = -cos(heading) * sin(pitch) + sin(heading) * sin(roll) * cos(pitch);
    rotateArray[8] = cos(roll) * cos(pitch);
	
    read_index_txt();

    for (size_t i = 0; i < index_vector.size(); i++)
    {
		cv::Mat frame;
		capture >> frame;

        std::string cloud_file = data_dir + "cloud/" + index_vector[i] + ".pcd";
        std::string label_file = data_dir + "label/" + index_vector[i] + ".txt";
        std::vector<std::string> label_vector;
        read_label_txt(label_file, label_vector);
        
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_file, *input_cloud) == -1)
        {
            printf("Couldn't read %s\n", cloud_file.c_str());
            return (-1);
        }
        int tempnum = input_cloud->points.size();
        LidarParsing(tempnum);

        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<float> buffer((float *)data, std::default_delete<float[]>());
        point2pixel(&data, rotateArray, translateArray, cameraArray, &length);
        buffer.reset((float *)data);
        float *img_uv = (float *)buffer.get(); 
        size_t points_size = length / 3;

		cv::Mat show_image;
        show_image = frame.clone();
        Image_display(show_image, img_uv, points_size);
		
		Parse_label(label_vector);
		
		float time = 0.1f;
        std::vector<Cluster> clusters;
		std::vector<Tracking> trackingList;
        for(int i = 0; i < Inference_result.size(); i++)
        {
			Cluster cluster;
			cluster.posX = Inference_result[i].xc;
			cluster.posY = Inference_result[i].yc;
			cluster.posZ = Inference_result[i].zc;
			cluster.width  = Inference_result[i].w;
			cluster.length = Inference_result[i].l;
			cluster.height = Inference_result[i].h;
			cluster.direct = Inference_result[i].yaw;
			cluster.type   = Inference_result[i].name;
            cluster.confidence = Inference_result[i].score;
			cluster.usedFlag = false;
			clusters.push_back(cluster);
        }
        long trackingStartTime = cv::getTickCount();          
        LidarTracking(clusters, trackingList, time);
        long trackingEndTime = cv::getTickCount();
        printf("object number = %d	tracking number = %d	", (int)Inference_result.size(), (int)trackingList.size());
        std::cout << "tracking time = " << (trackingEndTime - trackingStartTime) / cv::getTickFrequency() << std::endl;
		
		Visualize_pointcloud_results(trackingList);		

		Inference_result.clear();
        cloud->points.clear();
        if (viewer->wasStopped())
        {
        	return 0;
        }
    }
	index_vector.clear();
    return 0;
}

void Parse_label(std::vector<std::string> &object_vector)
{
	for (size_t i = 0; i < object_vector.size(); i++)
    {
		CloudResult temp_obj;
        std::vector<std::string> char_arr;
        SplitString(object_vector[i], char_arr, " ");
        //temp_obj.track_ID = atoi(char_arr[0].c_str());
        temp_obj.name = char_arr[1];
        temp_obj.xc = atof(char_arr[2].c_str());
        temp_obj.yc = atof(char_arr[3].c_str());
        temp_obj.zc = atof(char_arr[4].c_str());
        temp_obj.w  = atof(char_arr[5].c_str());
        temp_obj.l  = atof(char_arr[6].c_str());
        temp_obj.h  = atof(char_arr[7].c_str());
        temp_obj.yaw = atof(char_arr[8].c_str());

		std::random_device e;
        std::uniform_real_distribution<double> u(0.8, 1);
		temp_obj.score = u(e);

		Inference_result.push_back(temp_obj);
    }
}

void Visualize_pointcloud_results(std::vector<Tracking> &trackingList)
{
	std::vector<int> name_count(6, 0);
    for (int i = 0; i < (int)trackingList.size(); i++)
    {
        int ID = trackingList[i].ID;
        float centerX = trackingList[i].X(0);
        float centerY = trackingList[i].X(1);
        float centerZ = trackingList[i].posZ;
        float speedX = trackingList[i].X(2);
        float speedY = trackingList[i].X(3);
        float width = trackingList[i].width;
        float length = trackingList[i].length;
        float height = trackingList[i].height;
        float direct = trackingList[i].direct;
		float score = trackingList[i].confidence;
		string cls_name = trackingList[i].type;
		
		float heading = direct;
        Eigen::Quaternionf rotation(cos(heading / 2), 0, 0, sin(heading / 2));
        Eigen::Vector3f location(centerX, centerY, centerZ);
		
		float r, g, b;
        if (cls_name == "car")
        {
            r = 1.0;
            g = 0.0;
            b = 1.0;
            name_count[0]++;
        }
        else if (cls_name == "truck")
        {
            r = 0.0;
            g = 1.0;
            b = 0.0;
            name_count[1]++;
        }
        else if (cls_name == "bus")
        {
            r = 0.0;
            g = 0.0;
            b = 1.0;
            name_count[2]++;
        }
        else if (cls_name == "non_motor_vehicles")
        {
            r = 1.0;
            g = 1.0;
            b = 0.0;
            name_count[3]++;
        }
        else if (cls_name == "pedestrians")
        {
            r = 0.0;
            g = 1.0;
            b = 1.0;
            name_count[4]++;
        }
        else if (cls_name == "other_obstacles")
        {
            r = 1.0;
            g = 0.0;
            b = 0.0;
            name_count[5]++;
        }


        char strCube[100] = { 0 };
		sprintf(strCube, "cube%ld", flag++);
        viewer->addCube(location, rotation, length, width, height, strCube, 0);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, strCube);
        //viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, strCube);
        //viewer->setRepresentationToWireframeForAllActors();
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, strCube);
		viewer->setRepresentationToSurfaceForAllActors(); 

        pcl::PointXYZ pos_dist = pcl::PointXYZ(location(0) + length / 2, location(1), location(2) + height / 2);
		char strObj[100] = { 0 };
		double dDist = sqrt(location(0) * location(0) + location(1) * location(1));
        double velocity = sqrt(speedX * speedX + speedY * speedY);
        float angle = atan2f(location(0), location(1)) * 180 / CV_PI; //方位角
        if((centerX < 0 && centerY > 0) || (centerX < 0 && centerY < 0))
        {
            angle = angle + 360;
        }
        //sprintf(strObj, "heading_%0.2f_dist_%0.1fm_velo_%0.2fm/s_score_%.2f", heading, dDist, velocity, score);
		sprintf(strObj, "%0.1fm<%0.2fm/s>", dDist, velocity);		
		char strText[100] = { 0 };
		sprintf(strText, "dist%ld_", flag++);
		viewer->addText3D(strObj, pos_dist, 1.0, 255, 255, 255, strText);
    }

	//往每帧点云上添加注释性图标
    std::vector<int> posText = {10, 20};
	char carText[100] = { 0 };
	sprintf(carText, "car%ld", flag++);
    char carLegend[100] = {0};
    sprintf(carLegend, "car  %d", name_count[0]);
	viewer->addText(carLegend, posText[0], posText[1]+125, 20, 1.0, 0.0, 1.0, carText, 0); 

	char truckText[100] = { 0 };
	sprintf(truckText, "truck%ld", flag++);
    char truckLegend[100] = {0};
    sprintf(truckLegend, "truck  %d", name_count[1]);
	viewer->addText(truckLegend, posText[0], posText[1]+100, 20, 0.0, 1.0, 0.0, truckText, 0); 

    char busText[100] = { 0 };
	sprintf(busText, "bus%ld", flag++);
    char busLegend[100] = {0};
    sprintf(busLegend, "bus  %d", name_count[2]);
	viewer->addText(busLegend, posText[0], posText[1]+75, 20, 0.0, 0.0, 1.0, busText, 0);

    char non_motor_vehiclesText[100] = { 0 };
	sprintf(non_motor_vehiclesText, "non_motor_vehicles%ld", flag++);
    char non_motor_vehiclesLegend[100] = {0};
    sprintf(non_motor_vehiclesLegend, "non_motor_vehicles  %d", name_count[3]);
	viewer->addText(non_motor_vehiclesLegend, posText[0], posText[1]+50, 20, 1.0, 1.0, 0.0, non_motor_vehiclesText, 0);

    char pedestriansText[100] = { 0 };
	sprintf(pedestriansText, "pedestrians%ld", flag++);
    char pedestriansLegend[100] = {0};
    sprintf(pedestriansLegend, "pedestrians  %d", name_count[4]);
	viewer->addText(pedestriansLegend, posText[0], posText[1]+25, 20, 0.0, 1.0, 1.0, pedestriansText, 0);

    char other_obstaclesText[100] = { 0 };
	sprintf(other_obstaclesText, "other_obstacles%ld", flag++);
    char other_obstaclesLegend[100] = {0};
    sprintf(other_obstaclesLegend, "other_obstacles  %d", name_count[5]);
	viewer->addText(other_obstaclesLegend, posText[0], posText[1], 20, 1.0, 0.0, 0.0, other_obstaclesText, 0);
  
    viewer->updatePointCloud(cloud);
    viewer->addPointCloud(cloud);
    viewer->spinOnce(1);
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
}

void point2pixel(
                 void **data,
                 double *rotateArray, 
                 float *translateArray,
                 double *cameraArray,
                 unsigned int *length)
{
    unsigned int len = cloud->points.size() * 3;
    float *buffer = new float[len];
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        float point_cam[3] = {0};
        float point_lidar[3] = {cloud->points[i].x, cloud->points[i].y, cloud->points[i].z};

        point_cam[0] = translateArray[0];
	    point_cam[0] += point_lidar[0] * rotateArray[0];
	    point_cam[0] += point_lidar[1] * rotateArray[3];
	    point_cam[0] += point_lidar[2] * rotateArray[6];
	
	    point_cam[1] = translateArray[1];
	    point_cam[1] += point_lidar[0] * rotateArray[1];
	    point_cam[1] += point_lidar[1] * rotateArray[4];
	    point_cam[1] += point_lidar[2] * rotateArray[7];

	    point_cam[2] = translateArray[2];
	    point_cam[2] += point_lidar[0] * rotateArray[2];
	    point_cam[2] += point_lidar[1] * rotateArray[5];
	    point_cam[2] += point_lidar[2] * rotateArray[8];

        double tmpx = point_cam[0] / point_cam[2]; 
	    double tmpy = point_cam[1] / point_cam[2];
	    double px = 0.4 * cameraArray[0] * tmpx + cameraArray[2] + 45;
	    double py = 0.46 * cameraArray[4] * tmpy + cameraArray[5] - 20;

        buffer[3*i]   = px;
        buffer[3*i+1] = py;
        buffer[3*i+2] = point_cam[2];
    }
    *data = (void *)buffer;
    *length = len;
}

void Image_display(cv::Mat &show_image, float *img_uv, size_t &pixel_size)
{
    for(size_t i = 0; i < pixel_size; i++)
    {
        cv::Point2f pt;
        pt.x = img_uv[3 * i];
        pt.y = img_uv[3 * i + 1];
        float depth = (img_uv[3 * i + 2] / 100) * 255;

        float r, g, b; 
		if (depth <= 63)
        {
            r = 0;
            g = 254 - 4 * depth;
            b = 255;
        }
        else if (depth > 63 && depth <= 127)
        {
            r = 0;
            g = 4 * depth - 254;
            b = 510 - 4 * depth;
        }
        else if (depth > 127 && depth <= 191)
        {
            r = 4 * depth - 510;
            g = 255;
            b = 0;
        }
        else if (depth > 191 && depth <= 255)
        {
            r = 255;
            g = 1022 - 4 * depth;
            b = 0;
        }

        cv::circle(show_image, pt, 0.8, cv::Scalar(b, g, r), -1);    
    }
    cv::namedWindow("检测结果可视化", CV_WINDOW_NORMAL); //WINDOW_OPENGL
    cvResizeWindow("检测结果可视化", 640, 480);
    cv::imshow("检测结果可视化", show_image);
    cv::waitKey(1); 
}
