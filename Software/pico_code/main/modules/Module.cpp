/*
spi_test.cpp

Author: Tyler Bartunek

Test that we can receive and then send values over the SPI bus on the raspberry pi pico
*/

//#define and #include
#include "Module.h"


/****************************Main Function  *******************************/


Module::Module(const uint8_t identifier){

    //Start up SPI
    int rc = this->spi.init();
    hard_assert(rc == PICO_OK);

    //Set the mask for checksums: based on module ID
    MASK = identifier;

    //Initialize the LED to blink
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    //Start blinking timers
    burst_time = to_ms_since_boot(get_absolute_time());
    blink_time = to_ms_since_boot(get_absolute_time());
    blinks_done = 0;
    in_burst = false;
    led_on = false;

}

//Handle SPI transaction
short Module::Transfer(short data){

    //Blink according to error_code, which should be tied to status (do regardless)
    this->ErrorMessage();

    switch(status){

        case DISCONNECTED:

            {
            std::vector<uint8_t> sync_message = {0xFF, 0xAA, 0x55, 0xFF};
            std::vector<uint8_t> handshake_attempt;

            for (uint8_t h : sync_message){
                uint8_t incoming_byte = this->spi.transfer(h);
                handshake_attempt.push_back(incoming_byte);
            }

            if (handshake_attempt.size() >= 3 && handshake_attempt[1] + handshake_attempt[2] == 0xFF){
                status = TRANSMITTING;
            }

            }

            break;

        case TRANSMITTING:
            {
            //Get the outgoing message ready to send
            std::vector<uint8_t> sending = this->FrameMessage(data);
            //Prep the vector for receiving
            std::vector<uint8_t> receiving;
            //Perform the SPI transfer
            for (uint8_t m : sending){
                uint8_t incoming_byte = this->spi.transfer(m);
                receiving.push_back(incoming_byte);
            }
            //Parse the message
            short received = this->ParseMessage(receiving);

            return received;
            }

            break;

        case SUSPECT:

            break;

        default:

            break;
    }

}

//Frame the outgoing message
std::vector<uint8_t> Module::FrameMessage(short data){
    std::vector<uint8_t> message;
    message.push_back(MASK);

    // Use bitwise operations to break the 'short' into two bytes
    // Higher byte: Shift right by 8 bits and mask lowest 8 bits (optional mask but clear)
    unsigned char high_byte = (data >> 8) & 0xFF;
    // Lower byte: Mask just the lowest 8 bits
    unsigned char low_byte = data & 0xFF;

    // Add the high and low bytes of data to the vector
    message.push_back(high_byte);
    message.push_back(low_byte);

    std::vector<uint8_t> data_vec = {high_byte, low_byte};

    message.push_back(this->CRC_Generator(data_vec));

    return message; 
}


//Parse the incoming message
short Module::ParseMessage(const std::vector<uint8_t>& message){

    bool correct_header = this->Checksum(message[0]);
    bool correct_CRC = this->CRC_Checker(
                            std::vector<uint8_t>(message.begin() + 1, message.begin() + 3),
                            message[3]);

    if (correct_header && correct_CRC){
        status = TRANSMITTING;
        error_code = ALL_CLEAR;
        return (short(message[1]) << 8) | message[2];
    }
    else if (status == SUSPECT){
        status = DISCONNECTED;
        error_code = LOST_HOST;
        return -2;
    }
    else{
        status = SUSPECT;
        if (!correct_header){
            error_code = BAD_HEADER;
        }
        else if (!correct_CRC){
            error_code = BAD_CRC;
        }
        return -1;
    }

}

void Module::ErrorMessage(){

    switch(error_code){
        case ALL_CLEAR:
            //Bypass the Blink method, just keep it on.
            gpio_put(PICO_DEFAULT_LED_PIN, true);
            break;

        case BAD_HEADER:
            this->Blink(2, 250);
            break;

        case BAD_CRC:
            this->Blink(4, 250);
            break;

        case LOST_HOST:
        default:
            this -> Blink(2, 1000);
            break;
    }

}

void Module::Blink(const uint8_t num_blinks, const uint64_t delay_between_blinks){

    uint64_t now = to_ms_since_boot(get_absolute_time());

    /* Start a new burst once per second */
    if (!in_burst && (now - burst_time) >= 1000) {
        in_burst = true;
        blinks_done = 0;
        led_on = false;

        gpio_put(PICO_DEFAULT_LED_PIN, led_on);

        blink_time = now;
        burst_time = now;
        return;
    }

    /* If we're not in a burst, do nothing */
    if (!in_burst) {
        return;
    }

    /* Handle blinking inside the burst */
    if ((now - blink_time) >= delay_between_blinks) {
        led_on = !led_on;
        gpio_put(PICO_DEFAULT_LED_PIN, led_on);

        blink_time = now;

        /* Count only ON transitions as a blink */
        if (led_on) {
            blinks_done++;
            if (blinks_done >= num_blinks) {
                in_burst = false;
                led_on = false;
                gpio_put(PICO_DEFAULT_LED_PIN, led_on);
            }
        }
    }
}


bool Module::CRC_Checker(const std::vector<uint8_t>& payload, uint8_t received_crc){

    return CRC_Generator(payload) == received_crc;
}

uint8_t Module::CRC_Generator(const std::vector<uint8_t>& data){

    const uint16_t polynomial = 0x107; // 9-bit poly: x^8 + x^2 + x + 1 TODO: Replace
    uint16_t reg = 0;

    // Load data into register
    for (uint8_t byte : data) {
        reg ^= (uint16_t(byte) << 8);
        for (int i = 0; i < 8; ++i) {
            if (reg & 0x0100)
                reg = (reg << 1) ^ polynomial;
            else
                reg <<= 1;
        }
    }

    return uint8_t((reg >> 8) & 0xFF);

}

bool Module::Checksum(uint8_t header){

    return uint8_t(header + MASK) == 0xFF;

}