add_library(KeypointMatchingLib)
target_sources(KeypointMatchingLib #${PROJECT_NAME}_lib 
	PRIVATE 
		matchKeypoints.cpp
	PUBLIC
		matchKeypoints.hpp
)
target_include_directories(KeypointMatchingLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})