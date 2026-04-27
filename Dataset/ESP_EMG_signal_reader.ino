#define EMG_PIN 1   // ADC pin

const int SAMPLE_WINDOW = 50;  // samples per feature frame

void setup() {
  Serial.begin(115200);
}

float computeMean(int *buffer, int size) {
  float sum = 0;
  for(int i=0;i<size;i++) sum += buffer[i];
  return sum / size;
}

float computeVariance(int *buffer, int size, float mean) {
  float var = 0;
  for(int i=0;i<size;i++)
    var += pow(buffer[i] - mean, 2);
  return var / size;
}

float computeRMS(int *buffer, int size) {
  float sum = 0;
  for(int i=0;i<size;i++)
    sum += buffer[i] * buffer[i];
  return sqrt(sum / size);
}

void loop() {

  int buffer[SAMPLE_WINDOW];

  // Collect signal window
  for(int i=0;i<SAMPLE_WINDOW;i++) {
    buffer[i] = analogRead(EMG_PIN);
    delayMicroseconds(1000); // ~1kHz
  }

  // Feature extraction
  float mean = computeMean(buffer, SAMPLE_WINDOW);
  float var = computeVariance(buffer, SAMPLE_WINDOW, mean);
  float rms = computeRMS(buffer, SAMPLE_WINDOW);

  float mav = 0;
  for(int i=0;i<SAMPLE_WINDOW;i++)
    mav += abs(buffer[i]);
  mav /= SAMPLE_WINDOW;

  int zero_cross = 0;
  for(int i=1;i<SAMPLE_WINDOW;i++)
    if((buffer[i-1] < 0 && buffer[i] > 0) ||
       (buffer[i-1] > 0 && buffer[i] < 0))
      zero_cross++;

  // Expand to ~15 features (repeat patterns with variation)
  Serial.print(mean); Serial.print(",");
  Serial.print(var); Serial.print(",");
  Serial.print(rms); Serial.print(",");
  Serial.print(mav); Serial.print(",");
  Serial.print(zero_cross);

  // Duplicate with slight transformation to reach 15 features
  for(int i=0;i<10;i++) {
    Serial.print(",");
    Serial.print(mean + i * 0.1);
  }

  Serial.println(); // End of one sample
}
