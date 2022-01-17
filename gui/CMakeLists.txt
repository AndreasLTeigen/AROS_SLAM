add_library(GUILib)
target_sources(GUILib #${PROJECT_NAME}_lib 
	PRIVATE 
		pangolinInterface.cpp
		guiUtil.cpp
	PUBLIC
		pangolinInterface.hpp	
		guiUtil.hpp
)
target_include_directories(GUILib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})