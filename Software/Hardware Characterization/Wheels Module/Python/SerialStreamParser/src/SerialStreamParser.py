#Serial communication
import serial as ser
from time import perf_counter, sleep
from datetime import datetime
import struct

import os
import pandas as pd
import os.path as pth

#############################################################################################################################
##################################### Data Acquisition, Parsing, and CSV Storing ############################################
#############################################################################################################################

def ParseData(newline_data:bytes, outgoing_dict:dict[str,list], previous_values:list):

    """
    ParseData: Takes an incoming stream of data from the serial port and 
    parses it into the relevant categories to be parsed.

    Expecting 16 bytes + 1 stop byte, so 17 bytes total

    -4 bytes: Time, microseconds
    -4 bytes: Position x 100
    -4 bytes: Current x 100
    -2 bytes: replicate number
    -2 bytes: PWM
    -1 stop byte
    """

    categories = outgoing_dict.keys()
    output_dict = outgoing_dict.copy()
    mismatch = False

    STOP_BYTE = 0XF8 #11111000 is really unlikely to show up in either the time or pwm data

    if len(newline_data) != 17:
        mismatch = True

    elif newline_data[16] != STOP_BYTE:
        mismatch = True

    else:
        #Use previous values by default
        data_float_list = previous_values[:]

        try:
            candidate_tuple = struct.unpack('<Iiihhx', newline_data)
            candidates = [candidate_tuple[0] / 1000.0,  #Time (s)
                          candidate_tuple[1] / 100.0,  #Position, radians
                          candidate_tuple[2] / 100.0,  #Current, A
                          candidate_tuple[3],          #Run number
                          candidate_tuple[4]]          #PWM

            #Make sure that data is within expected ranges
            if (candidates[0] < 800) and (candidates[0] > 0): #Even the longest test takes only 1500 seconds
                data_float_list[0] = candidates[0]
            if abs(candidates[1]) < 2000:  #Position should never exceed 2000 radians
                data_float_list[1] = candidates[1]
            if abs(candidates[2]) < 1:  #Current should never exceed one ampere
                data_float_list[2] = candidates[2]
            if abs(candidates[3]) < 20: #Run number should never exceed 20
                data_float_list[3] = candidates[3]
            if abs(candidates[4]) < 256: #PWM command needs to be less than 255, with negative being opposite direction
                data_float_list[4] = candidates[4]

        except ValueError: #Something went wrong, skip this read
            pass

    for valueIdx, category in enumerate(categories):
        if mismatch:
            output_dict[category].append(previous_values[valueIdx])
        else:
            output_dict[category].append(data_float_list[valueIdx])
            previous_values[valueIdx] = data_float_list[valueIdx]
    
    return output_dict, previous_values

    
def StoreDataCSV(data_to_store:pd.DataFrame,filename:str, data_path = pth.join(pth.dirname(__file__), "data")) -> None:

    #Check if the filename string contains an extension, do this by checking for punctuation
    extension_check = filename.split(".")
    if len(extension_check) < 2:
        filename = filename + ".csv"
    elif len(extension_check) > 2:
        raise ValueError("filename should not contain more than one '.', as that can be confused with an extension")

    if not pth.exists(data_path): #If the path doesn't exist, make it up
        os.mkdir(data_path)

    data_to_store.to_csv(pth.join(data_path, filename), index = False)
    

def GatherAndStore(port_name:str, baud_rate, cutoff, filename):

    #Ready flag
    ready_flag = False

    #Initialize the dynamic arrays and dictionary for saving the data
    time_data = []
    PWM_data = []
    Pos_data = []
    Curr_data = []
    Run_data = []
    motor_data_dict = {"Time":time_data, "Position":Pos_data, "Current":Curr_data, "Run":Run_data, "PWM":PWM_data}
    previous_values = [0.0, 0, 0.0, 1, 0] #In case of a bad read

    #Open the port
    with ser.Serial(port_name, baud_rate, timeout = 0) as port:

        #Loop and collect data for some amount of time
        print("Port opened successfully")
        sleep(2) #Wait for Arduino to be ready
        port.reset_input_buffer()

        while not ready_flag:
            #Send trigger signal
            port.write(bytes("Ready\n",'utf-8'))
            #Await response
            readyMessage = port.readline().strip()
            if readyMessage == b'Ready':
                ready_flag = True

        #Start the timer
        port.read() #Discard a bad read
        startTime = perf_counter()
        currentTime = perf_counter() - startTime

        #While on the clock, collect data
        while currentTime < cutoff:

            port.write(b'\xFF')


            byte = port.read(1) #Grab the start byte
            if byte == b'\xFA':
                newline_data = port.read(17) #Read 17 bytes of data
                motor_data_dict, previous_values = ParseData(newline_data, motor_data_dict, previous_values)
                print("Time (s): {0:3.7f}, Run: {1:d}, Position: {2:3.4f}, Current: {3:3.4f}, PWM: {4:}".format(motor_data_dict['Time'][-1],
                                                                                                                 motor_data_dict['Run'][-1],motor_data_dict['Position'][-1], 
                                                                                                                 motor_data_dict["Current"][-1],motor_data_dict['PWM'][-1]))
            #Update the timer variable
            currentTime = perf_counter() - startTime


    #Package as data frame and save the data
    motor_data_df = pd.DataFrame(motor_data_dict)
    StoreDataCSV(motor_data_df, filename, data_path = pth.join(pth.dirname(__file__), "data/CRBD_Trials"))