add_library(TestLib)
target_sources(TestLib #${PROJECT_NAME}_lib 
	PRIVATE 
		matchingTest.cpp
		concurrencyTest.cpp
		testClassLink.cpp
		testClassNode.cpp
		#helloPangolin.cpp
		#pangolinMultiView.cpp
	PUBLIC
		matchingTest.hpp
		concurrencyTest.hpp
		testClassLink.hpp
		testClassNode.hpp
		#helloPangolin.hpp
		#pangolinMultiView.hpp
)

target_include_directories(TestLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})