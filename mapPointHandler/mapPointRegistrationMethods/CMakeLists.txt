add_library(MapPointRegistrationMethodsLib)
target_sources(MapPointRegistrationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		linearInclusiveMPReg.cpp
		depthGT.cpp
	PUBLIC
		linearInclusiveMPReg.hpp
		depthGT.hpp
)

target_include_directories(MapPointRegistrationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})