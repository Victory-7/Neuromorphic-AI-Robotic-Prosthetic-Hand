#include <Servo.h>

Servo fingerServo;

int servoPin = 5;   // Change if needed
int angle;

void setup() {
  Serial.begin(115200);
  fingerServo.attach(servoPin);

  Serial.println("Servo Calibration Started");
  Serial.println("Servo will increase by 10 degrees each step.");
}

void loop() {

  for(angle = 0; angle <= 180; angle += 10) {

    fingerServo.write(angle);   // Move servo
    Serial.print("Servo Angle: ");
    Serial.println(angle);      // Display angle on Serial Monitor

    delay(1500);  // Wait so you can observe finger movement
  }

  Serial.println("Sweep Finished. Restarting...");
  delay(5000);
}
