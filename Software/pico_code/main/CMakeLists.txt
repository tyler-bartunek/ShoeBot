# --- SPITools library ---
add_library(SPITools STATIC
    utils/SPITools.cpp
)

target_include_directories(SPITools PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(SPITools
    pico_stdlib
    hardware_spi
)

# --- Module base class ---
add_library(Module STATIC
    modules/Module.cpp
)

target_include_directories(Module PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

target_link_libraries(Module
    SPITools
    pico_stdlib
    pico_time
    hardware_spi
)

# --- BlinkEcho derived class ---
add_library(EchoDevice STATIC
    modules/EchoDevice.cpp
)

target_include_directories(EchoDevice PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/modules
)

# BlinkEcho depends on Module, which in turn depends on pico_stdlib
target_link_libraries(EchoDevice
    Module
    SPITools
    pico_stdlib
    pico_time
    hardware_spi
)

# --- Executable ---
add_executable(main
    src/main.cpp
)

target_link_libraries(main
    pico_stdlib
    hardware_spi
    SPITools
    Module
    EchoDevice
)

# create map/bin/hex file etc.
pico_add_extra_outputs(main)

