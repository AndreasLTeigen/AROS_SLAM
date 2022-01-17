add_library(TrackingLib)
target_sources(TrackingLib #${PROJECT_NAME}_lib 
	PRIVATE 
		tracking.cpp
	PUBLIC
		tracking.hpp
)

target_include_directories(TrackingLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})