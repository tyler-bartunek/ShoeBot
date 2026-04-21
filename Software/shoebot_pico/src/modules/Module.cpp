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

    //Initialize SYNC pin
    gpio_init(SYNC);
    gpio_set_dir(SYNC, GPIO_IN);

    //Start blinking timers
    error_code = LOST_HOST;
    burst_time = to_ms_since_boot(get_absolute_time());
    blink_time = to_ms_since_boot(get_absolute_time());
    blinks_done = 0;
    in_burst = true;
    led_on = false;

}

//Handle SPI transaction
short Module::Transfer(short data){

    switch(status){

        case DISCOVERY:

            {
            //Create sync and handshake attempt arrays
            uint8_t sync_message[MSG_LEN] = {};
            uint8_t handshake_attempt[MSG_LEN] = {};

            //Enforce transmission rules, sync message has 0-valued data
            this->FrameMessage(0, sync_message);

            //Send out the sync message
            this->spi.transfer(sync_message, handshake_attempt, MSG_LEN);

            if (this->IsConnectionEstablished(handshake_attempt)){
                PATH_ID = handshake_attempt[1] & 0x7;
                status = TRANSMITTING;
                return this->ParseMessage(handshake_attempt);
            }
            else{
                //Send to DISCONNECTED so we have a path to re-try.
                status = DISCONNECTED;
                prev_status = DISCOVERY;
            }

            }

            break;

        case TRANSMITTING:
            {
            //Get the outgoing message ready to send
            uint8_t sending[MSG_LEN] = {};
            this->FrameMessage(data, sending);
            //Prep the vector for receiving: force it to be the right size
            uint8_t receiving[MSG_LEN];

            //Perform the SPI transfer
            this->spi.transfer(sending, receiving, MSG_LEN);
            
            //Parse the message
            short received = this->ParseMessage(receiving);

            return received;
            }

            break;

        case SUSPECT:

            if ((prev_status == SUSPECT) && (missed_packets >= 3)){
                status = DISCONNECTED;
                error_code = LOST_HOST;
            }
            else{
                missed_packets++;
                status = TRANSMITTING; //Try again, without discovery protocol to see if we can get lucky
            }

            break;

        case DISCONNECTED:
        default:

            status = DISCOVERY;

            break;
    }

    return 0;

}


//Frame the outgoing message
void Module::FrameMessage(short data, uint8_t* message){

    if (this-> status == TRANSMITTING){
        //Construct the header, push the mask to host
        message[0] = 0xF0 | (PATH_ID & 0x7);
        message[1] = MASK;

        // Use bitwise operations to break the 'short' into two bytes
        // Higher byte: Shift right by 8 bits and mask lowest 8 bits (optional mask but clear)
        uint8_t high_byte = (data >> 8) & 0xFF;
        // Lower byte: Mask just the lowest 8 bits
        uint8_t low_byte = data & 0xFF;

        // Add the high and low bytes of data to the vector
        message[2] = high_byte;
        message[3] = low_byte;

        //Compute the checksum value based on first message length-2 values
        //this value because message length - (alignment + checksum)
        message[4] = this->Checksum(message, MSG_LEN - 2);

        //Append alignment byte
        message[5] = 0xBF;
    }
    else{
        //Push out a message of zeros with 0xFF checksum
        message[4] = 0xFF;
        message[5] = 0xBF;
    }
 
}


//Parse the incoming message
short Module::ParseMessage(const uint8_t* message){

    bool correct_path_id = (message[1] & 0x7) == PATH_ID;
    bool correct_mask = message[2] == MASK;
    uint8_t data_payload[MSG_LEN-2]= {message[1], message[2], message[3], message[4]};
    bool correct_checksum_value = this->ValidChecksum(data_payload, message[5], MSG_LEN);

    if (correct_path_id && correct_checksum_value && correct_mask){
        status = TRANSMITTING;
        error_code = ALL_CLEAR;
        prev_status = TRANSMITTING;
        missed_packets = 0;
        return (short(message[2]) << 8) | message[3];
    }
    else{
        status = SUSPECT;
        prev_status = SUSPECT;
        if (!correct_path_id){
            error_code = BAD_PATH_ID;
        }
        else if (!correct_checksum_value){
            error_code = BAD_CHECKSUM;
        }
        else if (!correct_mask){
            error_code = BAD_MASK;
        }
        return 0;
    }

}

void Module::ErrorMessage(){

    switch(error_code){
        case ALL_CLEAR:
            //Bypass the Blink method, just keep it on.
            gpio_put(PICO_DEFAULT_LED_PIN, true);
            break;

        case BAD_PATH_ID: //Have this cascade for now.

        case BAD_MASK:
            this->Blink(2, 250);
            break;

        case BAD_CHECKSUM:
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
    if (!in_burst && ((now - burst_time) >= 1000)) {
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

bool Module::IsConnectionEstablished(const uint8_t* handshake){

    bool eof_check = handshake[0] == 0xBF;
    //Strip the alignment and checksum bytes from checksum calc
    uint8_t data_payload[MSG_LEN-2] = {handshake[1], handshake[2], handshake[3], handshake[4]};
    bool checksum_valid = this -> ValidChecksum(data_payload, handshake[5], MSG_LEN-2);

    return eof_check && checksum_valid;

}


uint8_t Module::Checksum(const uint8_t* payload, size_t payload_len){

    uint8_t result = 0;

    for (uint8_t byte_idx = 0; byte_idx < payload_len; byte_idx++){
        result += payload[byte_idx];
    }

    return result;

}

bool Module::ValidChecksum(const uint8_t* payload, uint8_t sent_result, size_t payload_len){

    return this->Checksum(payload, payload_len) == sent_result;

}