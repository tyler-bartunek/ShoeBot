#pragma once

#include <Arduino.h>

//Misc helper functions
void shuffleArray(int arr[], int size);
bool HasElapsed(unsigned long &lastTime, unsigned long interval);
