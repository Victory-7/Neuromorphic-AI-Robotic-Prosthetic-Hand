#include <ESP32Servo.h>

#define NUM_FINGERS 5
#define NUM_GESTURES 26

Servo fingers[NUM_FINGERS];

int servoPins[NUM_FINGERS] = {_, _, _, _, _}; //pin numbers 

int gestureAngles[NUM_GESTURES][NUM_FINGERS] = {
  
  {00, 00, 00, 00, 00},  
  {00, 00, 00, 00, 00},  
  {00, 00, 00, 00, 00},  
  {00, 00, 00, 00, 00},  
  {00, 00, 00, 00, 00},  // 21 more

};

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < NUM_FINGERS; i++) {
    fingers[i].attach(servoPins[i]);
  }
}

void loop() {

  if (Serial.available()) {
    char gesture = Serial.read();

    if (gesture >= 'A' && gesture <= 'Z') {
      int index = gesture - 'A';
      moveGesture(index);
    }
  }
}

void moveGesture(int gestureIndex) {

  for (int i = 0; i < NUM_FINGERS; i++) {
    fingers[i].write(gestureAngles[gestureIndex][i]);
  }
}
