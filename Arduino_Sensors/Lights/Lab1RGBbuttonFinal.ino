const int red_light_pin = 11;
const int green_light_pin = 10;
const int blue_light_pin = 9;
const int button_pin = 7;
int counter = 0;

void setup() {
  // put your setup code here, to run once:
pinMode(red_light_pin, OUTPUT);
pinMode(green_light_pin, OUTPUT);
pinMode(blue_light_pin, OUTPUT);
pinMode(button_pin, INPUT);
Serial.begin(9600);
}

void loop() {
if (digitalRead(button_pin) == 1) {
  ++counter;
  delay(500);
  }

Serial.println(counter);
Serial.println(digitalRead(button_pin));
if (counter == 0) {RGB_color(0,0,0);
}
if (counter == 1) {RGB_color(255, 0, 0);
}
if (counter == 2) {RGB_color(0, 255, 0);
}
if (counter == 3) {RGB_color(0, 0, 255);
}
if (counter == 4) {RGB_color(255, 255, 125);
}
if (counter == 5) {RGB_color(255, 255, 0);
}
if (counter == 6) {(counter -= 5);
}

}

void RGB_color(int red_light_value, int green_light_value, int blue_light_value)
{
  analogWrite(red_light_pin, red_light_value);
  analogWrite(green_light_pin, green_light_value);
  analogWrite(blue_light_pin, blue_light_value);
}