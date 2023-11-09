void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  int reading;
  reading = analogRead(A0);
  float voltage;
  voltage = (reading)*(.0048875);
  float resistance;
  resistance = (1000*(1-voltage/5)*5)/voltage;
  Serial.println("Temperature is " + String((1/(1/294.9 + log(resistance/11630)/3010.8))-273.15) + "degrees celsius");
  int referencetempAnalog;
  referencetempAnalog = analogRead(A5);
  float referencetemp;
  referencetemp = ((referencetempAnalog * 4.88) - 500) *0.1;
  Serial.println("Reference Temp is " + String(referencetemp));
  delay(1000);

}
