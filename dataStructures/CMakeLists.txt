add_library(DataStructuresLib)
target_sources(DataStructuresLib #${PROJECT_NAME}_lib 
	PRIVATE 
		frameData.cpp
		pose.cpp
		keypoint.cpp
		map3D.cpp
		mapPoint.cpp
		match.cpp
		parametrization.cpp
	PUBLIC
		frameData.hpp
		pose.hpp
		keypoint.hpp
		map3D.cpp
		mapPoint.hpp
		match.hpp
		parametrization.hpp
)
target_include_directories(DataStructuresLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})