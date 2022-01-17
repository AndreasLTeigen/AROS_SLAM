add_library(MatchingMethodsLib)
target_sources(MatchingMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE 
		bruteForceMatching.cpp
	PUBLIC
		bruteForceMatching.hpp	
)
target_include_directories(MatchingMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})