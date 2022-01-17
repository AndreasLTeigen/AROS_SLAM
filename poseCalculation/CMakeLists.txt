add_library(PoseCalculationLib)
target_sources(PoseCalculationLib #${PROJECT_NAME}_lib 
	PRIVATE 
		poseCalculation.cpp
	PUBLIC
	poseCalculation.hpp
)
target_include_directories(PoseCalculationLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})