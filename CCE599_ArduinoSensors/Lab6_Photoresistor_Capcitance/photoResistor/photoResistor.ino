const int RGBred = 11;
const int RGBgreen = 10;
const int RGBblue = 9;
const int LEDred = 6;
const int LEDgreen = 5;
const int LEDblue = 4;
const int photoresistor = 0;
unsigned long lastTime = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(RGBred, OUTPUT);
  pinMode(RGBgreen, OUTPUT);
  pinMode(RGBblue, OUTPUT);
  pinMode(LEDred, OUTPUT);
  pinMode(LEDgreen, OUTPUT);
  pinMode(LEDblue, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  unsigned long currentTime = millis();

  int colorvalue;
  colorvalue = analogRead(photoresistor);

//RGB

  if (currentTime - lastTime < 3000) {
  analogWrite(RGBred, 255);
  analogWrite(RGBgreen, 0);
  analogWrite(RGBblue, 0);
  }
  if (currentTime - lastTime < 6000 && currentTime - lastTime > 3000) {
  analogWrite(RGBred, 0);
  analogWrite(RGBgreen, 255);
  analogWrite(RGBblue, 0);
  }
  if (currentTime - lastTime < 9000 && currentTime - lastTime > 6000) {
  analogWrite(RGBred, 0);
  analogWrite(RGBgreen, 0);
  analogWrite(RGBblue, 255);
  }
if (currentTime - lastTime > 9000) {
  lastTime = currentTime;
  }

  //print the sensor value
  Serial.println(colorvalue);

  if (colorvalue >= 134 && colorvalue <= 145) {
    digitalWrite(LEDred, HIGH);
    digitalWrite(LEDgreen, LOW);
    digitalWrite(LEDblue, LOW);
    Serial.println("This is red");
  }
  if (colorvalue >= 132  && colorvalue <= 133) {
    //added color
    Serial.println("This is orange");
  }
  if (colorvalue >= 120 && colorvalue <= 131) {
    digitalWrite(LEDred, LOW);
    digitalWrite(LEDgreen, HIGH);
    digitalWrite(LEDblue, LOW);
    Serial.println("This is green"); 
  }
  if (colorvalue >= 109  && colorvalue <= 119) {
    //added color
    Serial.println("This is cyan");
  }
  if (colorvalue >= 94  && colorvalue <= 100) {
    digitalWrite(LEDred, LOW);
    digitalWrite(LEDgreen, LOW);
    digitalWrite(LEDblue, HIGH);
    Serial.println("This is blue");
  }
  if (colorvalue >= 88  && colorvalue <= 94) {
    //added color
    Serial.println("This is violet");
  }
}
