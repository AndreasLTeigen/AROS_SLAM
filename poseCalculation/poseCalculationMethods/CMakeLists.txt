add_library(PoseCalculationMethodsLib)
target_sources(PoseCalculationMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		point5OutlierRejection.cpp
		nonLinReproOpt.cpp
		# GJET.cpp
		GJET2.cpp
		copyMotionPrior.cpp
		blockShift.cpp
	PUBLIC
		point5OutlierRejection.hpp
		nonLinReproOpt.hpp
		# GJET.hpp
		GJET2.hpp
		copyMotionPrior.hpp
		blockShift.hpp
)
target_include_directories(PoseCalculationMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})