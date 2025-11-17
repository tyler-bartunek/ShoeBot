#include "Utils.h"

//Array shuffling function using Fisher-Yates algo
void shuffleArray(int arr[], int size) {
  for (int i = size - 1; i > 0; i--) {
    int j = random(i + 1);  // pick a random index 0..i
    // swap arr[i] and arr[j]
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}


//Timer function
bool HasElapsed(unsigned long &lastTime, unsigned long interval) {
  unsigned long tslUpdate = millis() - lastTime;
  if (tslUpdate >= interval) {
    lastTime = millis();
    return true;
  } else {
    return false;
  }
}