cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11")

project(visual_point_cloud_with_3Dbbox)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED )


include_directories(
		${Boost_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${PCL_INCLUDE_DIRS}
		../include
)

link_directories(
    	${Boost_LIBRARY_DIRS}
    	${PCL_LIBRARY_DIRS}
    	${OpenCV_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS}) 

add_executable(visual_point_cloud_with_3Dbbox 
				visual_point_cloud_with_3Dbbox.cpp
)

target_link_libraries (visual_point_cloud_with_3Dbbox 
						${PCL_LIBRARIES} 
						${Boost_LIBRARIES} 
						${OpenCV_LIBS}
) 

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
