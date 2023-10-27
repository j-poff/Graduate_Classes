#include "pitches.h"


char character;
String inData;
String orginData;
const int ledPin = 12;
const int speakerPin = 8;
const int piezSensor = 0;
const int threshold = 200;
String preovilac;
const int readButton = 6;
String theword ="";

int melody[] = {NOTE_G6};
int noteDurations[] = {4,8};

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(speakerPin, OUTPUT);
  Serial.begin(9600);
  pinMode(readButton,INPUT);
}

void loop() {
  int sensorReading = analogRead(piezSensor);
  noTone(speakerPin);
  //Serial.println(sensorReading);

  if (sensorReading >= threshold) {
    unsigned long currentTime = millis();

    while (sensorReading >= threshold) {
      sensorReading = analogRead(piezSensor);
      // Serial.println("stuck in while loop, reading is " + String(sensorReading));
    }
    // now we are no longer touching it
    unsigned long elaspedTime = millis();
    unsigned long tapTime;
    tapTime = elaspedTime - currentTime;
    if (tapTime < 500){
      preovilac += char('*');
      Serial.println("this is what is stored " + String(preovilac));
      digitalWrite(ledPin, HIGH);
      tone(speakerPin, melody[0]);
      delay(300);

    } else {
      digitalWrite(ledPin, HIGH);
      preovilac += char('-');
      Serial.println("this is what is stored " + String(preovilac));
      int thisNote = 0;
      tone(speakerPin, melody[0]);
      delay(1000);
    }
  }
  noTone(speakerPin);
  digitalWrite(ledPin, LOW);
  
  if (digitalRead(readButton) == HIGH) {
    Serial.println("THE BUTTON IS ON");
    if (preovilac == String("*")){
        theword += 'E';
        preovilac.remove(0,1);
        delay(500);}
    else if (preovilac == String("****")){
        theword += 'H';
        preovilac.remove(0,4);
        delay(500);}
    else if (preovilac == String("*-**")){
        theword += 'L';
        preovilac.remove(0,4);
        delay(500);}
    else if (preovilac == String("---")){
        theword += 'O';
        preovilac.remove(0,3);
        delay(500);}
    else if (preovilac == String("***")){
        theword += 'S';
        preovilac.remove(0,3);
        delay(500);}
      else {
        preovilac.remove(0,100);
      }
    }
if (theword != String("")) {
Serial.println(theword);
}
  

  while (Serial.available() > 0) {

    char recieved = Serial.read(); //reads the input from hte serial monitor
    inData += recieved; //adds each individual charcter input in the serial monitor to the string inData
  

    if (recieved == '\n') {

      orginData = inData;
      inData.trim(); // removes any leading and trailing white space from the string
      if (inData.equals(String("h"))) { \
        Serial.println("H is ****");
      }
      else if (inData.equals(String("e"))) {
        Serial.println("E is *");
      }
      else if (inData.equals(String("l"))) {
        Serial.println("L is *-**");
      }
      else if (inData.equals(String("o"))) {
        Serial.println("O is ---");
      }
      else if (inData.equals(String("s"))) {
        Serial.println("S is ***");
      }
      else if (inData.equals(String("hello"))) {
        Serial.println("HELLO is **** * *-** *-** ---");
      }
      else if (inData.equals(String("sos"))) {
      Serial.println("SOS is *** --- ***");
      }
      orginData = "";
      inData = "";
    }
  }


}
