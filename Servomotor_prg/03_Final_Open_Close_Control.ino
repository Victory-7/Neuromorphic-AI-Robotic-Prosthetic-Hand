/*
  --------------------------------------------------------
  Finger Control
  File: 03_Final_Open_Close_Control.ino
  Board: ESP32 S3 Mini
  Servo: MG995
  Purpose: Control finger using calibrated open
           and close angles
  --------------------------------------------------------
*/

#include <Servo.h>

Servo fingerServo;
const int servoPin = 5;

// Update these values after calibration
const int openAngle  = 10;
const int closeAngle = 110;

void setup() {
  Serial.begin(115200);
  fingerServo.attach(servoPin);

  Serial.println("Finger Control System Initialized.");
}

void loop() {

  // Open Finger
  fingerServo.write(openAngle);
  Serial.println("Finger Opened.");
  delay(3000);

  // Close Finger
  fingerServo.write(closeAngle);
  Serial.println("Finger Closed.");
  delay(3000);
}
