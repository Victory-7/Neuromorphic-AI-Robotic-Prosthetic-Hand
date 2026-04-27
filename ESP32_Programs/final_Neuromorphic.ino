#include <ESP32Servo.h>

#define NUM_SERVOS 5
#define NUM_GESTURES 37

Servo servo[NUM_SERVOS];

// ✅ FIXED SAFE PINS (NO GPIO 9 & 10)
int servoPins[NUM_SERVOS] = {18, 9, 10, 19, 21};

// Your inversion setup
bool invertServo[NUM_SERVOS] = {  
  false,  // Thumb
  false,  // Index
  true,   // Middle
  false,  // Ring
  true    // Pinky
};

// =========================
// GESTURE MATRIX (0–36)
// =========================
int gestureAngles[NUM_GESTURES][NUM_SERVOS] = {

  {0,140,140,140,140},    // 0 - Fist
  {100,0,0,0,0},          // 1 - Open
  {60,0,0,0,0},           // 2 - C
  {100,140,140,140,140},  // 3 - E
  {0,100,0,0,0},          // 4 - OK
  {100,140,140,140,0},    // 5 - I
  {0,0,140,140,140},      // 6 - L
  {100,100,100,100,100},  // 7 - O
  {100,140,140,140,140},  // 8 - S
  {100,0,0,120,120},      // 9 - U
  {100,0,0,140,140},      // 10 - V
  {100,0,0,0,140},        // 11 - W
  {0,140,140,140,0},      // 12 - Y

  {50,90,120,120,120},    // 13 - D
  {50,90,120,120,120},    // 14 - G
  {120,30,30,120,120},    // 15 - H
  {60,30,30,120,120},     // 16 - K
  {90,30,30,120,120},     // 17 - P
  {120,30,30,120,120},    // 18 - R

  {100,140,140,140,0},    // 19 - J
  {100,110,110,110,110},  // 20 - M
  {120,120,120,120,120},  // 21
  {100,110,110,140,140},  // 22 - N
  {120,120,120,120,120},  // 23
  {100,110,140,140,140},  // 24 - Q
  {120,120,120,120,120},  // 25
  {0,110,140,140,140},    // 26 - T 
  {100,110,140,140,140},  // 27 - X
  {100,0,140,140,140},    // 28 - Z

  {0,0,0,0,0},            // 29 - Neutral
  {0,140,140,140,140},    // 30 - Thumbs Up
  {100,0,140,140,140},    // 31 - Index Point
  {100,140,0,0,0},        // 32 - Okay sign 
  {100,140,140,140,140},  // 33 - Peace
  {0,0,0,0,0},            // 34 - Open hand
  {100,140,140,140,140},  // 35 - Fist
  {80,120,120,120,120}    // 36 - Light grip
};

// =========================
// ANGLE CORRECTION
// =========================
int correctAngle(int angle, bool invert) {
  if (invert) return 180 - angle;
  return angle;
}

// =========================
// APPLY GESTURE
// =========================
void applyGesture(int g) {

  if (g < 0 || g >= NUM_GESTURES) {
    Serial.println("❌ Invalid gesture");
    return;
  }

  Serial.print("✅ Executing Gesture: ");
  Serial.println(g);

  for (int i = 0; i < NUM_SERVOS; i++) {
    int finalAngle = correctAngle(gestureAngles[g][i], invertServo[i]);
    servo[i].write(finalAngle);
  }
}

// =========================
// SETUP
// =========================
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < NUM_SERVOS; i++) {
    servo[i].attach(servoPins[i]);
  }

  delay(1000);
  Serial.println("🤖 System Ready...");
}

// =========================
// LOOP (READ NUMBERS ONLY)
// =========================
void loop() {

  static int lastGesture = -1;

  if (Serial.available()) {

    int gesture = Serial.parseInt();  // ✅ expects numbers now

    // Flush leftover characters
    while (Serial.available()) Serial.read();

    Serial.print("📥 Received: ");
    Serial.println(gesture);

    if (gesture != lastGesture) {
      applyGesture(gesture);
      lastGesture = gesture;
    }
  }
}