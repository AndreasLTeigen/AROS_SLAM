add_library(FramePreprocessMethodsLib)
target_sources(FramePreprocessMethodsLib #${PROJECT_NAME}_lib 
	PRIVATE
		autocorrelation.cpp
		noise.cpp
		blur.cpp
		fft.cpp
		homomorphicFilter.cpp
	PUBLIC
		autocorrelation.hpp
		noise.hpp
		blur.hpp
		fft.hpp
		homomorphicFilter.hpp
)
target_include_directories(FramePreprocessMethodsLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
