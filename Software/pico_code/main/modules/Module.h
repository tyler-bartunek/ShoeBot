#pragma once


#include "../utils/SPITools.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include <cstdint>
#include <vector>

class Module{

    protected:

        //State machine keeping track of connection status
        enum ConnectionStates {DISCONNECTED, TRANSMITTING, SUSPECT};
        ConnectionStates status = DISCONNECTED;

        //SPI communication tools
        SPI_Bus spi;

        //Identifier
        uint8_t MASK;

        //Blinking error codes
        enum ERRORS {ALL_CLEAR, BAD_HEADER, BAD_CRC, LOST_HOST};
        ERRORS error_code;
        uint64_t burst_time;
        uint64_t blink_time;
        bool led_on;
        bool in_burst;
        uint8_t blinks_done;

    public:

        explicit Module(const uint8_t identifier);
        //Handle SPI transaction
        short Transfer(short data);
        virtual ~Module() = default;

    protected:
        //Frame the outgoing message
        std::vector<uint8_t> FrameMessage(short data);

        //Parse the incoming message
        short ParseMessage(const std::vector<uint8_t>& message);

        bool CRC_Checker(const std::vector<uint8_t>& payload, uint8_t received_crc);

        uint8_t CRC_Generator(const std::vector<uint8_t>& data);

        bool Checksum(uint8_t header);

        void ErrorMessage();

        void Blink(const uint8_t num_blinks, const uint64_t delay_between_blinks);

};

// class Module {
// protected:
//     enum ConnectionStates { DISCONNECTED, TRANSMITTING, SUSPECT };
//     ConnectionStates status = DISCONNECTED;

//     SPI_Bus spi;
//     uint8_t MASK;

//     // Hooks
//     virtual void onConnected() {}
//     virtual void onDisconnected() {}
//     virtual void onValidMessage(short data) {}
//     virtual void onInvalidMessage() {}

// public:
//     explicit Module(uint8_t mask);
//     virtual ~Module() = default;

//     short Transfer(short data);

// protected:
//     std::vector<uint8_t> FrameMessage(short data);
//     short ParseMessage(const std::vector<uint8_t>& message);
//     bool CRC_Checker(const std::vector<uint8_t>& payload,
//                      uint8_t received_crc);
//     uint8_t CRC_Generator(const std::vector<uint8_t>& data);
//     bool Checksum(uint8_t header);
// };