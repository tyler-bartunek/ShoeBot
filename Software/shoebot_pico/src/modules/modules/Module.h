#pragma once


#include "SPITools.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include <cstdint>

#define MSG_LEN 6

class Module{

    protected:

        //SYNC pin
        uint8_t SYNC = 22;

        //State machine keeping track of connection status
        enum ConnectionStates {DISCONNECTED, TRANSMITTING, SUSPECT};
        ConnectionStates status = DISCONNECTED;
        ConnectionStates prev_status = DISCONNECTED;
        uint8_t missed_packets = 0;

        //SPI communication tools
        SPI_Bus spi;

        //Module identifier, path connection
        uint8_t MASK;
        uint8_t PATH_ID;

        //Blinking error codes
        enum ERRORS {ALL_CLEAR, BAD_PATH_ID, BAD_CHECKSUM, BAD_MASK, LOST_HOST};
        ERRORS error_code;
        uint64_t burst_time;
        uint64_t blink_time;
        bool led_on;
        bool in_burst;
        uint8_t blinks_done;

    public:

        explicit Module(const uint8_t identifier);
        
        //Define within inherited classes to obtain desired behavior
        virtual void run() = 0;
        virtual void sync_callback() {}
        
        virtual ~Module() = default;

    protected:

        //Handle SPI transaction
        [[nodiscard]] short Transfer(short data);

        //Frame the outgoing message
        void FrameMessage(short data, uint8_t* ou);

        //Parse the incoming message
        [[nodiscard]] short ParseMessage(const uint8_t* message);

        bool IsConnectionEstablished(const uint8_t* handshake);

        uint8_t Checksum(const uint8_t* payload, size_t payload_len);

        bool ValidChecksum(const uint8_t* payload, uint8_t sent_result, size_t payload_len);

        void ErrorMessage();

        void Blink(const uint8_t num_blinks, const uint64_t delay_between_blinks);

};