add_library(PoseCalculationMethodsLib)
target_sources(PoseCalculationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		point5OutlierRejection.cpp
	PUBLIC
		point5OutlierRejection.hpp
)
target_include_directories(PoseCalculationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})