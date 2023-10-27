
int soft_pot = 0;
int leaving = 0;
int entering = 0;

void setup() {
  Serial.begin(9600);

}

void loop() {
int sensorValue;
sensorValue = analogRead(soft_pot); // read the value from the soft potentiometer
Serial.println(sensorValue);

if(sensorValue > 980) { //if the sensor indicates a high value, it means someone is leaving the room and stepped on the edge of the potentiometer.
  ++leaving; //add a number to the leaving variable
  while (sensorValue > 50) { //until they are detected at the end of the strip, they have not left yet
    sensorValue = analogRead(soft_pot);
    Serial.println(sensorValue);
  }
  delay(2000); //delay 2 seconds for them to clear the sensor and for it to go back to normal
}

if(sensorValue < 50) { //if the sensor indicates a low value, it means someone is entering the room and stepped on the other edge of the potentiometer.
  ++entering; //add a number to the entering variable
  while (sensorValue < 980) { // until they are detected at the end of the strip, they have not entered yet
    sensorValue = analogRead(soft_pot);
    Serial.println(sensorValue);
  }
  delay(2000);//delay 2 seconds for them to clear the sensor and for it to go back to normal
}

Serial.println("Left: " + String(leaving));
Serial.println("Entered: " + String(entering));
Serial.println("People in the room: " + String(entering-leaving)); //print how many people left, entered, and how many are in the room now

}