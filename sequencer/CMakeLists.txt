add_library(SequencerLib)
target_sources(SequencerLib #${PROJECT_NAME}_lib 
	PRIVATE 
		sequencer.cpp
		sequencer2.cpp
	PUBLIC
		sequencer.hpp
		sequencer2.hpp
)
target_include_directories(SequencerLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
