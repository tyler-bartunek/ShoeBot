/*
Sketch name: Romeo_MotorCharacterize.ino
Author: Tyler Bartunek
Purpose/Function: To collect data on the relationship between PWM command and speed.

Update as of September 19 2025: Moved some helper functions into header files and added use of templates
*/

//#define and #include
#include "Encoder.h"
#include "Wire.h"
#include "ADS1115_WE.h"
#include "SerialComms.h"
#include "Trajectory.h"
#include "Utils.h"

//Motor constants
#define GEAR_RATIO 19.0      //Reduction ratio of 1:19 (1 turn of output to 19 turns of input)
#define COUNTS_PER_REV 44.0  //11 pulses per revolution, quadrature encoding

//Hardware pins
#define MOTOR_IN1 6
#define MOTOR_IN2 9
#define ENC_A 2
#define ENC_B 3

//I2C Address:ADDR pin left floating
#define ADC_I2C_ADDR 0x48

//Encoder object
Encoder MotorEncoder = Encoder(ENC_A, ENC_B);

//ADC Object
ADS1115_WE adc = ADS1115_WE(ADC_I2C_ADDR);

//Commands being sent
int availableCommands[8] = { 31, 63, 95, 127, 159, 191, 223, 255 };

//Timing variables
unsigned long startTime = 0;  //Time that the setup completed, ms


void setup() {
  //Serial: set baud rate
  Serial.begin(230400);

  //I2C: Get wire going
  Wire.begin();

  //Set the direction pin mode
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);

  //Set the counts appropriately
  MotorEncoder.write(0);

  //Start the ADC, get settings configured
  if (!adc.init()) {
    Serial.println("Fault state of ADC");  //Hopefully this doesn't awaken something
  }

  //Set appropriate gain, fastest rate, continuous mode to compare A0 and A1
  adc.setVoltageRange_mV(ADS1115_RANGE_0256); //Theoretically we could see as much as 556 mV, realistically more like 78
  //Options: 6144, 4096, 2048, 1024, 0512, 0256
  adc.setMeasureMode(ADS1115_CONTINUOUS);
  adc.setCompareChannels_nonblock(ADS1115_COMP_0_1); //Advised if we want non-blocking
  adc.setConvRate(ADS1115_860_SPS);

  bool started = false;  //Flag for keeping track of if we've started data acquisition

  //Await input from the Python script
  while (!started) {
    if (Serial.available() > 4) {
      String readyMessage = Serial.readStringUntil('\n');
      readyMessage.trim();
      if (readyMessage == "Ready") {
        Serial.println("Ready");
        started = true;
      }
    }
  }

  //Start the clock(s)
  startTime = millis();
}



void loop() {
  // put your main code here, to run repeatedly:

  //Trajectory and data structs initialized
  static Trajectory traj;
  static TestData data;

  //Flags
  static bool sending = true;
  static bool receiving = false;

  //Get the motor command, drive the motor (optional)
  data.motorCommand = GetTrajectory(traj, data);
  //Comment if not driving motor
  DriveMotor(data.motorCommand);

  //Get the current time and the time since the previous reading
  data.time = (millis() - startTime);

  if (HasElapsed(data.time, 2)) {  //Get and print readings every 2 ms

    //Read the encoder value: radians
    float reading = 2 * PI * MotorEncoder.read() / (COUNTS_PER_REV * GEAR_RATIO);
    data.position = long(reading * 100);

    //Read the current
    float currentFloat = adc.getResult_V() / .2609938;  //Dividing by shunt resistance
    data.current = long(currentFloat * 100);

    //We'll start with a Serial print, move on to a Serial write if necessary.
    if (sending) {
      SendData(data, 0xFA, 0xF8);
      sending = false;
      receiving = true;
    }
    if (receiving) {
      if (CheckReceived(0xFF)) {
        sending = true;
        receiving = false;
      }
    }
  }
}