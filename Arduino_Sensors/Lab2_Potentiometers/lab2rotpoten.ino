
int soft_pot = 0;

// single side positive, opp neg in line,

void setup() {
  // put your setup code here, to run once
Serial.begin(9600);
}

void loop() {
int sensorValue;
sensorValue = analogRead(soft_pot);
int angle;
angle = (sensorValue * .2629) - 140.5;
Serial.println("The angle is " + String(angle));
delay(500);
}