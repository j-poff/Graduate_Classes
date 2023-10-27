const int capacitor_sensor = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int bitsRead;
  bitsRead = analogRead(capacitor_sensor); //read the sensor into a variable bitsRead
  float voltage;
  voltage = bitsRead * .004888; // We must convert from bits (0-1023) to our voltage of 5, *(5/1023)
  Serial.println("The voltage read is " + String(voltage)); //report the voltage reading
  float waterHeight;
  waterHeight = (voltage * 17.298) + 0.1254; // use the transfer function to relate voltage to water height
  Serial.println("The water height is " + String(waterHeight) + "cm"); //report water height
  delay(1000); //delay 1 second for readability
}