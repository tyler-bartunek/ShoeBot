#Add the utils library
add_library(utils STATIC SPITools.cpp)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(utils PUBLIC hardware_spi pico_stdlib)


