#Create the Modules library
add_library(modules STATIC Module.cpp EchoDevice.cpp)

target_include_directories(modules PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(modules PUBLIC utils)


