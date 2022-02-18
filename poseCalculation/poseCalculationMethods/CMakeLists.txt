add_library(PoseCalculationMethodsLib)
target_sources(PoseCalculationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		point5OutlierRejection.cpp
		copyMotionPrior.cpp
	PUBLIC
		point5OutlierRejection.hpp
		copyMotionPrior.hpp
)
target_include_directories(PoseCalculationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})