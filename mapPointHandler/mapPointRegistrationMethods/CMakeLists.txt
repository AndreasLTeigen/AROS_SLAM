add_library(MapPointRegistrationMethodsLib)
target_sources(MapPointRegistrationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		linearInclusiveMPReg.cpp
	PUBLIC
		linearInclusiveMPReg.hpp
)

target_include_directories(MapPointRegistrationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})