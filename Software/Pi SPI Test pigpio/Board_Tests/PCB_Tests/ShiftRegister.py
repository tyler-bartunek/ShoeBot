import pigpio

class ShiftRegister:
    
    def __init__(self, pi, data_pin:int, latch_pin:int, sck_pin:int, oe_pin:int):
        
        self.data_pin = data_pin
        self.latch_pin = latch_pin
        self.sck_pin = sck_pin
        self.oe_pin = oe_pin
        
        #Configure pigpio object to control gpio pins
        self.pi = pi
        self.connect_pins()

        #Disable outputs by default until our first write
        self.pi.write(self.oe_pin, 1)

        #index variable for data bitarray and flag to mark when message is sent
        self.bit_index = 0
        self.done_sending = False
        
        
    def write(self, data:int):
        
        #Set data attribute for use in callback function
        self.data_list = self.to_bitarray(data)
        
        #Pull latch low for shift
        self.pi.write(self.latch_pin, 0)
        
        #Send data bit by bit to the shift register
        for bit in self.data_list:
            self.pi.write(self.data_pin, bit)
            self.pi.write(self.sck_pin, 1)
            self.pi.write(self.sck_pin, 0)

        #Enable outputs, shift values to storage register and outputs
        self.pi.write(self.latch_pin, 1)
        self.pi.write(self.oe_pin, 0)

    
    def connect_pins(self) -> None:
        
        self.pi.set_mode(self.data_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.latch_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.sck_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.oe_pin, pigpio.OUTPUT)

        return None
            
    # def rising_edge_callback(self, gpio, level, tick):
        
    #     if self.bit_index < len(self.data_list):
    #         self.pi.write(self.data_pin, self.data_list[self.bit_index])
    #         print("Writing {}".format(self.data_list[self.bit_index]))
    #         self.bit_index += 1
    #     else:
    #         self.done_sending = True
    #         self.pi.write(self.latch_pin, 1)      
             
    def to_bitarray(self, data:int) -> list:
        
        bit_array = []
        remainder = data
        
        for i in range(7,-1,-1):
            
            current_bit = 2 ** i
            if remainder >= current_bit:
                bit_array.append(1)
                remainder -= current_bit
            else:
                bit_array.append(0)

        return bit_array