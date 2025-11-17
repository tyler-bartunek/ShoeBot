#include "SerialComms.h"

//Check if received matches sync byte
bool CheckReceived(unsigned char syncByte) {
  if (Serial.available() > 0) {
    unsigned char incoming = Serial.read();
    return (incoming == syncByte);
  }
}

void SendData(TestData &data, unsigned char startByte, unsigned char stopByte) {
  if (Serial.availableForWrite() >= data.dataPacketByteCount() + 2) {
    Serial.write(startByte);
    publish(data.time);
    publish(data.position);
    publish(data.current);
    publish(data.runNumber);
    publish(data.motorCommand);
    Serial.write(stopByte);
  }
}