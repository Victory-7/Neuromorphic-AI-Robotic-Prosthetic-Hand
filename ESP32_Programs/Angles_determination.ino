#include <ESP32Servo.h>

Servo s;

int pin = 18;        
int initialPos = 0;  

void setup() {
  s.attach(pin);

  // Move to initial position safely
  s.write(initialPos);
  delay(2000);
}

void loop() {

  for (int cycle = 0; cycle < 4; cycle++) {

    // OPEN → CLOSE
    for (int angle = 0; angle <= 180; angle += 10) {
      s.write(angle);
      delay(800);   // adjust speed here
    }

    delay(1000);

    // CLOSE → OPEN
    for (int angle = 180; angle >= 0; angle -= 10) {
      s.write(angle);
      delay(800);
    }

    delay(1000);
  }

  // Return to initial position
  s.write(initialPos);

  // Stop program completely
  while (true);
}
