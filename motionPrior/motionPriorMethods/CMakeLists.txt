add_library(MotionPriorMethodsLib)
target_sources(MotionPriorMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		constant.cpp
		groundTruth.cpp
	PUBLIC
		constant.hpp
		groundTruth.hpp
)


target_include_directories(MotionPriorMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})