/*
SPITools.cpp

Author: Tyler Bartunek

Flesh out the methods that belong to the SPI_Bus object
*/

//#define and #include
#include "SPITools.h"

#define BUF_LEN 1

//Constructor
SPI_Bus::SPI_Bus(){}

int SPI_Bus::init(){

        spi_init(spi0, 100);                                            // value of 100 is baud rate, ignored in peripheral mode
        spi_set_format(spi0, 8 * BUF_LEN, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST); // Only sending one byte in this test anyway
        spi_set_slave(spi0, true);                                      // Set the device to be in peripheral mode

        // Configure CS, COPI, CIPO, and SCK pins
        gpio_set_function(CS, GPIO_FUNC_SPI);
        gpio_disable_pulls(CS);
        gpio_set_function(CIPO, GPIO_FUNC_SPI);
        gpio_set_function(COPI, GPIO_FUNC_SPI);
        gpio_set_function(SCK, GPIO_FUNC_SPI);

        // Make the SPI pins available to picotool
        bi_decl(bi_4pins_with_func(CIPO, COPI, SCK, CS, (uint32_t)GPIO_FUNC_SPI)); //RX, TX, SCK, CS

        //Set OUTPUT_BUFFER for initial transfer
        OUTPUT_BUFFER = 0;

        return PICO_OK; // Defined in stdlib as 0 (pico-sdk/src/common/pico_base_headers/include/pico/error.h on github)
}


uint8_t SPI_Bus::transfer(uint8_t data){
    //TODO: Edit this method to send out the data vector
    //Receives and returns data to the controller over the SPI bus
    OUTPUT_BUFFER = data;
    spi_write_read_blocking(spi0, &OUTPUT_BUFFER, &INPUT_BUFFER, BUF_LEN);

    return SPI_Bus::INPUT_BUFFER;
}
