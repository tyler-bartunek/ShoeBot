#Add my subdirectories
add_subdirectory(utils)
add_subdirectory(modules)

#Add the main.cpp file as an executable
add_executable(shoebot_module main.cpp)

#Link my libraries
target_link_libraries(shoebot_module PRIVATE modules pico_stdlib)

pico_enable_stdio_usb(shoebot_module 1)
pico_enable_stdio_uart(shoebot_module 0)

pico_add_extra_outputs(shoebot_module)

