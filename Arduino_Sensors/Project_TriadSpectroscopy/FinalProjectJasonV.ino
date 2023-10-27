/*
  Read the 18 channels of spectral light over I2C using the Spectral Triad
  By: Nathan Seidle
  SparkFun Electronics
  Date: October 25th, 2018
  License: MIT. See license file for more information but you can
  basically do whatever you want with this code.

  This example takes all 18 readings and blinks the illumination LEDs 
  as it goes. We recommend you point the Triad away from your eyes, the LEDs are *bright*.
  
  Feel like supporting open source hardware?
  Buy a board from SparkFun! https://www.sparkfun.com/products/15050

  Hardware Connections:
  Plug a Qwiic cable into the Spectral Triad and a BlackBoard
  If you don't have a platform with a Qwiic connection use the SparkFun Qwiic Breadboard Jumper (https://www.sparkfun.com/products/14425)
  Open the serial monitor at 115200 baud to see the output
*/

#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
AS7265X sensor;

#include <Wire.h>
#define AVG_SIZE 10
float avgReadings[AVG_SIZE];
byte avgReadingSpot = 0;

void setup()
{
  Serial.begin(115200);
  Serial.println("AS7265x Spectral Triad Example");

  Serial.println("Point the Triad away and press a key to begin with illumination...");
  while (Serial.available() == false)
  {
  }              //Do nothing while we wait for user to press a key
  Serial.read(); //Throw away the user's button

  if (sensor.begin() == false)
  {
    Serial.println("Sensor does not appear to be connected. Please check wiring. Freezing...");
    while (1)
      ;
  }

  sensor.disableIndicator(); //Turn off the blue status LED

  Serial.println("A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L");
}

void loop()
{
  sensor.takeMeasurementsWithBulb(); //This is a hard wait while all 18 channels are measured
  float channel7;
  channel7 = sensor.getCalibratedG(); //read channel 7
  
 avgReadings[avgReadingSpot] = channel7;
  if(avgReadingSpot == AVG_SIZE){
      avgReadingSpot = 0;
    

    float avgReading = 0;
    for (int x = 0 ; x < AVG_SIZE ; x++)
      avgReading += avgReadings[x];
    avgReading /= AVG_SIZE;

  Serial.println();
  if (channel7 > 75 && channel7 < 95) {
  Serial.println("Cooking oil"); }
  else if (channel7 > 33 && channel7 < 55) {
  Serial.println("Motor oil"); }
  else if (channel7 > 15 && channel7 < 30) {
  Serial.println("Dirt");}
  else if (channel7 > 110 && channel7 < 130) {
  Serial.println("Tap Water");}
  else {Serial.println("Recalibrate parameters");}
  }

  avgReadingSpot = avgReadingSpot + 1;

  
  
}
