int GetTrajectory(Trajectory &traj, TestData &data) {

  //Initialize output
  static int command = 0;
  static int newCommand = 0;
  int maxReplicates = 5;
  int sizeCommandArray = sizeof(availableCommands) / sizeof(availableCommands[0]);

  switch (traj.state) {

    case START:
      
      shuffleArray(availableCommands, sizeCommandArray);
      newCommand = availableCommands[0];
      traj.lastChangeTime = millis();
      traj.state = MOVE;

      break;

    case MOVE:

      command = newCommand;

      if (HasElapsed(traj.lastChangeTime, 15000)) {  //Wait 15 seconds before changing command value
        newCommand = GetNextCommand(traj, data, availableCommands, sizeCommandArray);
        traj.state = RESET;
      }

      break;

    case RESET:

      command = 0;

      if (HasElapsed(traj.lastChangeTime, 5000)){

        if (!traj.finished){
          MotorEncoder.write(0);
          traj.state = MOVE;
        }
        else{
          if (data.runNumber <= maxReplicates) {
          traj.finished = false;
          MotorEncoder.write(0);
          traj.state = START;
        } else {
          traj.state = STOP;
        }
        }
      }

      break;

    case STOP:
      command = 0;
  }

  return command;
}