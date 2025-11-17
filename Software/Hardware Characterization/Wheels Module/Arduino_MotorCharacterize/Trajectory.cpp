
#include "Trajectory.h"

int GetNextCommand(Trajectory &traj, TestData &data, int *commandArray, int arraySize) {
  if (IsValidIndex(traj, arraySize)) {  //If we are within valid command index values, fetch the next command
    traj.commandIndex += 1;
    return commandArray[traj.commandIndex];
  } else {
    UpdateRunNumber(traj, data);
    return 0;
  }
}

bool IsValidIndex(Trajectory &traj, int arraySize) {
  return (traj.commandIndex < (arraySize - 1));
}

void UpdateRunNumber(Trajectory &traj, TestData &data) {
  data.runNumber += 1;
  traj.finished = true;
  traj.commandIndex = 0;
}