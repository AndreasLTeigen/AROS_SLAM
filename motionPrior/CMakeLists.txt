add_library(MotionPriorLib)
target_sources(MotionPriorLib #${PROJECT_NAME}_lib 
	PRIVATE 
		motionPrior.cpp
	PUBLIC
		motionPrior.hpp
)


target_include_directories(MotionPriorLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})