add_library(KeypointExtractionMethodsLib)
target_sources(KeypointExtractionMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		orb.cpp
		bucketing.cpp
		extraction_gt.cpp
	PUBLIC
		orb.hpp
		bucketing.hpp
		extraction_gt.hpp	
)
target_include_directories(KeypointExtractionMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
