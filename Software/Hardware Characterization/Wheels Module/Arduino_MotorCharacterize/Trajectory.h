#pragma once

#include "SerialComms.h"
#include <Arduino.h>

//typedef our state machine
typedef enum {
  START,
  MOVE,
  RESET,
  STOP
} CharacterizationStates;

//Trajectory struct
struct Trajectory {
  int commandIndex = 0;
  bool finished = false;
  unsigned long lastChangeTime = 0;
  CharacterizationStates state = START;
};

int GetNextCommand(Trajectory &traj, TestData &data, int *commandArray, int arraySize);
bool IsValidIndex(Trajectory &traj, int arraySize);
void UpdateRunNumber(Trajectory &traj, TestData &data);