add_library(SequencerLib)
target_sources(SequencerLib #${PROJECT_NAME}_lib 
	PRIVATE 
		sequencer.cpp
	PUBLIC
		sequencer.hpp
)
target_include_directories(SequencerLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
