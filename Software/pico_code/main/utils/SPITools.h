#pragma once

#include "pico/stdlib.h" //Standard library for the raspberry pi pico
#include "hardware/spi.h"
#include "pico/binary_info.h"
#include <cstdint>

#define BUF_LEN 1

class SPI_Bus{

protected:

    //constexpr indicates this can be evaluated at compile time
    static constexpr uint8_t SCK = 18;
    static constexpr uint8_t COPI = 19; //TX
    static constexpr uint8_t CIPO = 20; //RX
    static constexpr uint8_t CS = 21;

    uint8_t INPUT_BUFFER, OUTPUT_BUFFER;

public:

    //Constructor
    SPI_Bus();

    //init function/method
    int init();

    //method for conducting data transfers
    uint8_t transfer(uint8_t data);

};

// struct SPI_Bus{

//     //Indicates this can be evaluated at compile time
//     static constexpr uint8_t SCK = 18;
//     static constexpr uint8_t COPI = 19; //TX
//     static constexpr uint8_t CIPO = 20; //RX
//     static constexpr uint8_t CS = 21;

//     //Initialize buffers
//     inline static uint8_t INPUT_BUFFER, OUTPUT_BUFFER;

//     //init function/method
//     int init(){
//         spi_init(spi0, 100);                                            // value of 100 is baud rate, ignored in peripheral mode
//         spi_set_format(spi0, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST); // Only sending one byte in this test anyway
//         spi_set_slave(spi0, true);                                      // Set the device to be in peripheral mode

//         // Configure CS, COPI, CIPO, and SCK pins
//         gpio_set_function(CS, GPIO_FUNC_SPI);
//         gpio_disable_pulls(CS);
//         gpio_set_function(CIPO, GPIO_FUNC_SPI);
//         gpio_set_function(COPI, GPIO_FUNC_SPI);
//         gpio_set_function(SCK, GPIO_FUNC_SPI);

//         // Make the SPI pins available to picotool
//         bi_decl(bi_4pins_with_func(CIPO, COPI, SCK, CS, (uint32_t)GPIO_FUNC_SPI)); //RX, TX, SCK, CS

//         //Set OUTPUT_BUFFER for initial transfer
//         OUTPUT_BUFFER = 0;

//         return PICO_OK; // Defined in stdlib as 0 (pico-sdk/src/common/pico_base_headers/include/pico/error.h on github)
//     }

//     bool IsConnected(unsigned char handshake_confirmation){
//         // Determines if the device is connected to the controller and receiving messages

//         if (INPUT_BUFFER == handshake_confirmation){
//             //If we establish connection, let the other device know by echoing
//             OUTPUT_BUFFER = INPUT_BUFFER;
//             spi_write_read_blocking(spi0, &OUTPUT_BUFFER, &INPUT_BUFFER, BUF_LEN);
//             return true;
//         }
//         else{
//             return false;
//         }
//     }

//     void ScanForConnection(){
//         //If so, read it. Assuming docs saying src and dst refer to source and destination
//         spi_read_blocking(spi0, 0, &INPUT_BUFFER, BUF_LEN);
//     }

//     void Echo(){
//         //Receives and returns data to the controller over the SPI bus
//         spi_write_read_blocking(spi0, &OUTPUT_BUFFER, &INPUT_BUFFER, BUF_LEN);
//         OUTPUT_BUFFER = INPUT_BUFFER; //Make sure that the input received is sent out on next pass
//     }
// };