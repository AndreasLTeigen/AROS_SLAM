add_library(UtilLib)
target_sources(UtilLib #${PROJECT_NAME}_lib 
	PRIVATE 
		util.cpp
	PUBLIC
		util.hpp
)
target_include_directories(UtilLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})