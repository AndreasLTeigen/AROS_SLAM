add_library(KeypointExtractionMethodsLib)
target_sources(KeypointExtractionMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		bucketing.cpp
		extraction_gt.cpp
	PUBLIC
		bucketing.hpp
		extraction_gt.hpp	
)
target_include_directories(KeypointExtractionMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
