#pragma once

#include <Arduino.h>

//Template for publishing data of integer types (unsigned long, long, int)
template <typename I>
void publish(I dataToSend){
  const int num_bytes = sizeof(I);
  union{
    I value;
    unsigned char b[num_bytes];
  }dataPacket;

  dataPacket.value = dataToSend;
  Serial.write(dataPacket.b, num_bytes);
};

//TestData struct
struct TestData {
  unsigned long time = 0;
  long position = 0;
  long current = 0;
  int runNumber = 1;
  int motorCommand = 0;

  static constexpr unsigned char dataPacketByteCount(void){
    return sizeof(time) + sizeof(position) + sizeof(current) + sizeof(runNumber) + sizeof(motorCommand);
  }
};

//Serial comm helper functions
bool CheckReceived(unsigned char syncByte);
void SendData(TestData &data, unsigned char startByte, unsigned char stopByte);