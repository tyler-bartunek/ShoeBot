/*
BlinkEcho.cpp

Author: Tyler Bartunek

Blinks when disconnected, echos data with the LED steady when connected.
*/

//#define and #include
#include "EchoDevice.h"

//Constructor
EchoDevice::EchoDevice() : Module(0xF0){}

void EchoDevice::run(){

    this->Echo();

}


void EchoDevice::Echo(){

    //Transfer data
    static short data_to_send = 0;
    short data_received = this->Transfer(data_to_send);

    //Ensure that data_to_send becomes what we received
    data_to_send = data_received;

}
