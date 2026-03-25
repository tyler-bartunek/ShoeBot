#Imports
import os.path as pth
from os import mkdir
import pigpio
import random

#From adjacent files
from BOARD_GLOBALS import *
from Test_Parameter_Globals import *
from ShiftRegister import ShiftRegister
from SPI_Board import SPIHub, FalseBoard


####################################################################################################################
############################################# Logging Helper Functions #############################################
####################################################################################################################
def SetupLogging(rep:int, loc:str, seq:int, data_path:str = None):

    if not data_path:
        data_path = pth.join(pth.dirname(__file__), "data")

    #If the path doesnt exist already, make it
    MkdirIfPathNotFound(data_path)

    filename = "Rep_{}_Loc_{}_Seq_{}.csv".format(rep, loc, seq)
    full_path = pth.join(data_path, filename)

    f = open(full_path, 'w')

    f.write("Rep,Loc,Seq,Freq,Sent,Received\n")

    return f

def MkdirIfPathNotFound(path:str) -> None:

    if not pth.exists(path):
        mkdir(path)

    return None

def PackageResults(f, sent, received, rep:int, loc:str, seq:int, freq:float):

    for idx, val in enumerate(sent):

        f.write("{0:},{1:},{2:},{3:},{4:},{5:}\n".format(rep, loc, seq, freq, val, received[idx]))

    return f

####################################################################################################################
################################################ Location Testing ##################################################
####################################################################################################################
def TestLocation(spi_hub:SPIHub, sequence_list:list, Test, num_vals:int = 1000, rep:int = None, loc:str = None, data_path:str = None, logging:bool = False) -> None:

    #Set the frequency
    for seq in sequence_list:
        #Setup logging for this location if loc specified
        if loc and logging:
            f = SetupLogging(rep, loc, seq, data_path)

        for freq in sequence_dict[seq]:
                        
            #Enable the bus
            spi_hub.enable_bus(CHANNEL, freq)

            #Execute test: Need to pass frequency and number of iterations as args
            sent, received = Test(freq, num_vals)

            if logging:
                f_new = PackageResults(f, sent, received, rep, loc, seq, freq)

            #Disable hub for next frequency
            spi_hub.disable_bus()

        f.close()
    
    return None


def EchoTest(spi_hub, location:str, freq:int, num_iters = 1000):

    sent = []
    received = []

    for val in range(num_iters):
        #Pick a random vaue to send
        test_value = random.randint(0,255)

        sent.append(test_value)

        #Send the random value twice, log second value received
        for i in range(2):
            received_bytes = spi_hub.transfer(location, test_value.to_bytes(1, byteorder = "big"), CHANNEL, freq, testing = True)
            received_array = bytearray(received_bytes)
            received_value = received_array[0]

        received.append(received_value)

    if len(sent) != len(received):
        raise("Mismatch in length of sent and received arrays")
    else:
        return sent, received

####################################################################################################################
################################################# Hardware Setup ###################################################
####################################################################################################################
def connect_pigpio():
    #Creates the pigpio object, sets it up and returns it
    pi = pigpio.pi()
    if not pi.connected:
        exit()

    return pi

####################################################################################################################
################################################ Data Simulator ####################################################
#################################################################################################################### 
def EchoMismatchTest(board_sim:FalseBoard):

    sent, received = EchoTest(board_sim, 'RL')

def PicoCommTest(hub:SPIHub, connection_point:str):

    for i in range(256):

        received = hub.transfer(connection_point, i.to_bytes(1, byteorder = "big"), CHANNEL, SYNC_RATE, testing = True)
        received_array = bytearray(received)
        received_value = received_array[0]

        print("Sent {}, received {}".format(i, received_value))

####################################################################################################################
################################################### Full Test ######################################################
####################################################################################################################

def TheBigKahuna(hub:SPIHub, save_dir:str):
    """
    Game plan:
        1. For each cell of design:
            a. Establish connection
            b. Send and receive values for comparison

    Seq 0 will be skipped in proper analysis, though exploratory could be of interest.
    """

    reps, sequences = list(range(5,7)), list(range(6))

    #Get the location
    for rep in reps:
        print("Beginning rep {}".format(rep))

        rep_folder = pth.join(save_dir, 'Rep_{}'.format(rep))
        MkdirIfPathNotFound(rep_folder)

        for loc in replicate_dict[rep]: 

            print("Connect pico to location {}".format(loc))

            rx = 0 

            #Set frequency low as possible, send 0xFF
            print("Scanning...")
            while rx != b'\xFF': 
                rx = hub.transfer(loc, b'\xFF', CHANNEL, SYNC_RATE, testing = True)

            #Established connection, disable hub to reset freq
            print("Connection obtained, running tests...\n")
            hub.disable_bus()
        
            #Run echo test over all frequencies
            test_type = lambda freq, num_vals: EchoTest(hub, loc, freq, num_iters = num_vals)
            TestLocation(hub, sequences, test_type, rep = rep, loc = loc, logging = True, data_path = rep_folder)
            print("done\n")

    print("Tests complete")

####################################################################################################################
##################################################### Main #########################################################
####################################################################################################################

def main():

    #Initialize the pigpio daemon
    pi = connect_pigpio()
    
    #Initialize the ShiftRegister object
    shift = ShiftRegister(pi, DATA, LATCH, SHIFT_CLK, OE)

    #Set the default cs pin to be an ouput
    pi.set_mode(CS, pigpio.OUTPUT)
    
    #Initialize the Board class
    hub = SPIHub(pi, shift)

    #Create a folder for the experimental data
    data_folder = pth.join(pth.dirname(__file__), "data/refined_tests")
    MkdirIfPathNotFound(data_folder)

    #If we're testing filesaving and such
    testing = False
    if testing:
        #Create a folder for functionality testing data
        test_data = pth.join(data_folder, "code_functionality_tests")
        MkdirIfPathNotFound(test_data)

    logCreationTest = False #Passed test
    TestEchoLengthMismatch = False #Passed? Some debugging necessary but seems to work now
    bigTestFalseBoard = False #Passed after some type casting
    testPicoConnection = False
    bigTest = True

    #Run through the test as defined in globals
    try:
        
        #Test that the filename is created correctly
        if logCreationTest:
            test_file = SetupLogging(0, "LogCreationTest", 0, data_path = test_data)
            test_file.close()
        #Test that echo works
        if TestEchoLengthMismatch:
            #Create the false board
            echo_test_board = FalseBoard(pi, shift)
            EchoMismatchTest(echo_test_board)
        if bigTestFalseBoard:
            #Create the false board
            echo_test_board = FalseBoard(pi, shift)

            big_echo_folder = pth.join(test_data, 'false_board_echo_data')
            MkdirIfPathNotFound(big_echo_folder)

            #Run the big test with the false board
            print("Testing with the false board")
            TheBigKahuna(echo_test_board, big_echo_folder)
        
        #Test that the pico's SPI code is functioning properly
        if testPicoConnection:

            #Location to test
            connection_point = 'XX'

            #Set frequency low as possible, send 0xFF
            rx = 0
            print("Scanning...")
            while rx != b'\xFF':  
                rx = hub.transfer(connection_point, b'\xFF', CHANNEL, SYNC_RATE, testing = True)

            print("Connection obtained, running pico comm test...\n")

            PicoCommTest(hub, connection_point)
            print('\n Complete')

        #Running through the whole test once ready
        if bigTest:
            TheBigKahuna(hub, data_folder)
    
    except KeyboardInterrupt:
        print("Process terminated by user")
        pi.spi_close(h_spi)
        #break


if __name__ == "__main__":
    
    #Execute the code
    main()
