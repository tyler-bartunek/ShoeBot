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

    //Blink according to error_code, which should be tied to status (do regardless)
    this->ErrorMessage();
    this->Echo();

}


void EchoDevice::Echo(){

    //Initialize data_to_send and data_received
    static short data_to_send = 0;
    short data_received;

    //Check if we are transmitting or if we've lost connection. If we have then we want to know about it
    if (status == TRANSMITTING)
        data_received = this->Transfer(data_to_send);
    else if (status != DISCOVERY)
        data_received = -626; //Value unlikely to appear by accident in transfer, means bad 

    //Ensure that data_to_send becomes what we received
    data_to_send = data_received;

}
