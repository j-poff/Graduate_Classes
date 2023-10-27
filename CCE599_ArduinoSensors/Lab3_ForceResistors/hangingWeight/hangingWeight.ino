#include <Wire.h>

#include "SparkFun_Qwicc_Scale_NAU7802_Arduino_library.h"

NAU7802 myScale;

void setup() {
Serial.begin(9600);
Serial.println("Qwiic Scale Example");

Wire.begin();

if (myScale.begin() == false) {
  Serial.println("Scale not detected.Please check wiring. Freezing...");
  while (1);
}
Serial.println("Scale detected!");
}

void loop() {
  if(myScale.avaliable() == true) {
    long currentReading = myScale.getReading();
    Serial.print("Reading: ");
    Serial.println(currentReading);
    delay(1000);
  }

}
