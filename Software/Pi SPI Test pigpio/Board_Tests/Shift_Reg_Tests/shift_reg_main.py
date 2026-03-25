#Imports
import pigpio

#From adjacent files
from SHIFT_GLOBALS import *
from ShiftRegister import *

def connect_pigpio():
        
    pi = pigpio.pi()
    if not pi.connected:
        exit()

    return pi

def main():

    #Initialize the pigpio daemon
    pi = connect_pigpio()

    #Last 0 is for flags
    # h_spi = pi.spi_open(CHANNEL, RATE, 0)
    
    #Initialize the ShiftRegister object
    shift = ShiftRegister(pi, DATA, LATCH, SPI0_SCLK)
    
    data_to_write = 31 #Random number, initially

    try:
        shift.write(data_to_write)
    
    except KeyboardInterrupt:
        print("Process terminated by user")
        pi.spi_close(h_spi)
        #break


if __name__ == "__main__":
    
    #Execute the code
    #try:
    main()

    #If finished or unexpectedly stopped, close the pigpio daemon
    #finally:
        #pigpio.stop()