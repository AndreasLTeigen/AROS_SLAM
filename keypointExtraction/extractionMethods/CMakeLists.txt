add_library(KeypointExtractionMethodsLib)
target_sources(KeypointExtractionMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		orb.cpp
		orbOS.cpp
		# orb_test.cpp
		bucketing.cpp
		extraction_gt.cpp
		blockFeatures.cpp
	PUBLIC
		orb.hpp
		orbOS.hpp
		# orb_test.hpp
		bucketing.hpp
		extraction_gt.hpp
		blockFeatures.hpp
)
target_include_directories(KeypointExtractionMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
