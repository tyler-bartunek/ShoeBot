#Basic functionality
import pigpio
import random #Specifically for the false board tests

#Custom tools
from ShiftRegister import *
from BOARD_GLOBALS import *

class SPIHub:
	
	def __init__(self, pi, register:ShiftRegister):

		self.pi = pi
		self.reg = register

		#Define spi handle as None so enable_bus and disable_bus logic works correctly
		self.h_spi = None
		

	def enable_bus(self, channel, rate) -> None:

		if not self.h_spi:
			#Last zero is for flags
			self.h_spi = self.pi.spi_open(channel, int(rate), 0)


	def disable_bus(self) -> None:

		if self.h_spi:
			self.pi.spi_close(self.h_spi)
		
		self.h_spi = None


	def toggle_cs(self, line_id:str, testing:bool = False, default_cs = CS) -> None:

		line_dict = {'RL':0b01111111, 'CL':0b10111111, 'FL':0b11011111,
		             'FR':0b11101111, 'CR':0b11110111, 'RR':0b11111011,
					 'XX':0xFF}
		alt_keys_numeric = list(range(7))
		
		#need to add logic to check if alternate keys are being used
		line_to_select = line_dict[line_id]

		self.reg.write(line_to_select)

		if testing and (line_to_select == 0xFF):
			self.pi.write(default_cs, 0)
		else:
			self.pi.write(default_cs, 1)


	def transfer(self, line_id:str, data:int, channel, rate, testing:bool = False, default_cs = CS):

		#open bus if not open, select line, send message, line high

		#Select line
		self.toggle_cs(line_id, testing = testing, default_cs = default_cs)

		#Enable the bus if it isn't already active
		self.enable_bus(channel, rate)

		(count, rx) = self.pi.spi_xfer(self.h_spi, data)

		#Pulse CS high at end of transaction
		self.toggle_cs('XX', testing = False, default_cs = default_cs)

		return rx

#False Board for Simulation/Debugging Purposes
class FalseBoard(SPIHub):

	def __init__(self, pi, register:ShiftRegister):
		
		self.pi = pi
		self.reg = register

		#Define spi handle as None so enable_bus and disable_bus logic works correctly
		self.h_spi = None

	def transfer(self, line_id:str, data:int, channel, rate):

		#Added this to make sure that the sync-up works for peripheral detection
		if data == b'\xFF':
			return data
		else:
			return random.randint(0,255)