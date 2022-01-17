add_library(MapPointHandlerLib)
target_sources(MapPointHandlerLib #${PROJECT_NAME}_lib 
	PRIVATE 
		mapPointRegistration.cpp
		mapPointCulling.cpp
	PUBLIC
        mapPointRegistration.hpp
		mapPointCulling.hpp
)


target_include_directories(MapPointHandlerLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})