add_library(FramePreprocessorLib)
target_sources(FramePreprocessorLib #${PROJECT_NAME}_lib 
	PRIVATE 
		framePreprocessor.cpp
	PUBLIC
		framePreprocessor.hpp
)
target_include_directories(FramePreprocessorLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})