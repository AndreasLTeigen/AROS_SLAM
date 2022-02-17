add_library(KeypointExtractionLib)
target_sources(KeypointExtractionLib #${PROJECT_NAME}_lib 
	PRIVATE 
		keypointExtraction.cpp
	PUBLIC
		keypointExtraction.hpp	
)
target_include_directories(KeypointExtractionLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
