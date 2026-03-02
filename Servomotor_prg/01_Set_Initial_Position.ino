/*
  --------------------------------------------------------
  Finger Control
  File: 01_Set_Initial_Position.ino
  Board: ESP32 S3 Mini
  Servo: MG995
  Purpose: Set servo to 0° before attaching tendon string
  --------------------------------------------------------
*/

#include <Servo.h>

Servo fingerServo;        // Create servo object
const int servoPin = 5;   // GPIO connected to servo signal

void setup() {
  Serial.begin(115200);   // Start serial communication (optional debugging)

  fingerServo.attach(servoPin);   // Attach servo to pin
  fingerServo.write(0);           // Move servo to 0° reference position

  Serial.println("Servo set to 0 degrees (Reference Position).");
}

void loop() {
  // No continuous action required
}
