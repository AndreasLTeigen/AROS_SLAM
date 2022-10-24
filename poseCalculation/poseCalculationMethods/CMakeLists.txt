add_library(PoseCalculationMethodsLib)
target_sources(PoseCalculationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		point5OutlierRejection.cpp
		nonLinReproOpt.cpp
		GJET.cpp
		copyMotionPrior.cpp
		blockShift.cpp
	PUBLIC
		point5OutlierRejection.hpp
		nonLinReproOpt.hpp
		GJET.hpp
		copyMotionPrior.hpp
		blockShift.hpp
)
target_include_directories(PoseCalculationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})