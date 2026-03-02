/*
  --------------------------------------------------------
  Finger Control
  File: 02_Calibration_Test.ino
  Board: ESP32 S3 Mini
  Servo: MG995
  Purpose: Gradually test servo angles to determine
           finger movement limits
  --------------------------------------------------------
*/

#include <Servo.h>

Servo fingerServo;
const int servoPin = 5;

void setup() {
  Serial.begin(115200);
  fingerServo.attach(servoPin);

  Serial.println("Starting Calibration Test...");
}

void loop() {

  for (int angle = 0; angle <= 130; angle += 5) {

    fingerServo.write(angle);
    Serial.print("Current Angle: ");
    Serial.println(angle);

    delay(2000);  // Observe finger movement
  }

  Serial.println("Calibration Cycle Complete.");
  delay(5000);  // Wait before repeating
}
