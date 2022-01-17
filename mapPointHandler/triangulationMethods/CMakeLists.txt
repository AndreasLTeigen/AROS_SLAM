add_library(TriangulationMethodsLib)
target_sources(TriangulationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		triangulateMany.cpp
	PUBLIC
		triangulateMany.hpp
)

target_include_directories(TriangulationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})