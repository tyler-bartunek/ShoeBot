void DriveMotor(int command) {
  /*
Drives motor according to DRV8871 datasheet,
with 0 representing assumed braking and 255 representing 
full speed ahead. 

Note that we toggle the direction by pulling one of
the inputs high. This is so when the other input goes
high, we enter the braking regime; when the other input
goes low we move in the intended direction. This is the
recommended approach by the datasheet for PWM operation.
*/
  if (command > 0) {  //Forward
    digitalWrite(MOTOR_IN1, HIGH);
    analogWrite(MOTOR_IN2, 255-abs(command)); //Motor driver does opposite, forward when low and break when high
  } else if (command == 0) {  //Brake
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, HIGH);
  } else {  //Reverse
    digitalWrite(MOTOR_IN2, HIGH);
    analogWrite(MOTOR_IN1, 255-abs(command));
  }
}
