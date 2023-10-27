const int FRS_PIN = A0;
const float radius = .05 //defining our variable constants, radius of sensor
const float waterSWeight = 9.807 //kN/m^3


void setup() {
  Serial.begin(9600);
  pinMode(FRS_PIN, INPUT);
}

void loop() {
float x;
x = analogRead(FRS_PIN); //read the voltage
float water_height;
float mass;
mass = pow(2.718, ((x+86.533)/145.29)); //calculate mass from voltage with inverse transfer equation
water_height = (((mass/1000)*9.81)/(3.14159*pow(radius, 2)*waterSWeight)); // calculate water height with hydrostatic force equation
Serial.println(water_height); //report the water height
delay(500); //delay to give time to read

}



