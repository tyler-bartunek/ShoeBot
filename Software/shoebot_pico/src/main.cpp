/*
main.cpp

Author: Tyler Bartunek

Ultimately going to be the main project code since it uses the actual framework
of the project to send and receive messages.

Module options:
    1. BlinkEcho: blinks onboard LED when disconnected, LED solid and echos messages received via SPI.
    2. Wheels (coming soon): Two configs (A and B), moves a Mecanum wheel according to received instructions
*/

//#define and #include:
#include "EchoDevice.h"
// #include "Wheels.h"


/****************************Main Function  *******************************/
int main(void){
    // Uncomment for debugging, and #include <iostream> in relevant class definition header file
    // stdio_init_all();  // initializes USB or UART stdout depending on config, for debug purposes
    // sleep_ms(3000);

    EchoDevice pico_device = EchoDevice();
    //Wheels pico_device = Wheels();

    //Once connected, continuously read the SPI input buffer and write to the transmit buffer
    while(true){
       pico_device.run();
    }

    return 0;
}