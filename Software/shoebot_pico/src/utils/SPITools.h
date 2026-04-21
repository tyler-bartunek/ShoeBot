#pragma once

#include "pico/stdlib.h" //Standard library for the raspberry pi pico
#include "hardware/spi.h"
#include "pico/binary_info.h"
#include <cstdint>

class SPI_Bus{

protected:

    //Define pins for SPI connections
    uint8_t SCK = 18;
    uint8_t COPI = 19; //TX
    uint8_t CIPO = 20; //RX
    uint8_t CS = 21;


public:

    //Constructor: 
    SPI_Bus(const uint8_t cs, const uint8_t copi, const uint8_t cipo, const uint8_t sck);

    //Default Constructor
    SPI_Bus();

    //init function/method
    int init();

    //method for conducting data transfers
    void transfer(const uint8_t* data, uint8_t* rx, size_t BUF_LEN);

    //Destructor
    ~SPI_Bus() = default;

};