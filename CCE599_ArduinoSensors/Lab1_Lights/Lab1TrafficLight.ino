const int PED_GREEN = 13;
const int PED_RED = 12;
const int LED_RED = 11;
const int LED_YELLOW = 10;
const int LED_GREEN = 9;
const int button_pin = 7;
const int reed_switch = 5;
int pedest = 0;
unsigned int old_time = 0;
unsigned int current_time = 0;
int track = 0;
int greenlight = 0;

void setup() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(button_pin, INPUT);
  pinMode(PED_RED, OUTPUT);
  pinMode(PED_GREEN, OUTPUT);
  pinMode(reed_switch, INPUT);
  Serial.begin(9600);
}

void loop() {
current_time = millis();
Serial.println(pedest);
Serial.println(track);

if (digitalRead(reed_switch) == HIGH) { 
  if (greenlight == 1) { //if the reed switch detects a car while the light is green
    while (digitalRead(reed_switch) == HIGH) { // while the car is detected
      digitalWrite(LED_GREEN, HIGH); // keep the light green. once it is off, the cycle will reset and light will be yellow.
      digitalWrite(LED_YELLOW, LOW);
      digitalWrite(LED_RED, LOW);
    }
  }
}

if (digitalRead(button_pin) == HIGH) { // tracks if the pedestrian button is pressed
  (pedest = 1);
}

if (track == 0) { // this is the normal cycle with no pedestrian
  if (current_time - old_time < 4000) { //the yellow cycle for 4 seconds
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, HIGH);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
  }
  if (current_time - old_time > 4000 && current_time - old_time < 11000) { // the red cycle for 7 seconds
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
  }
  if (current_time - old_time > 11000 && current_time - old_time < 23000) { // the green cycle for 12 seconds
  (greenlight = 1);
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
  }
  if (current_time - old_time > 23000) { // resets the time when it goes past 23 seconds.
    if (pedest == 1) { //if someone has pressed the signal
      (greenlight = 0);
      (track = 1); // use the pedestrian signal next time
      (pedest = 0); //reset
      (old_time = current_time); // resest the time
      Serial.println(pedest);
      Serial.println(track);
    } else { // no one has pressed the signal
      (greenlight = 0);
      (track = 0); // keep the regular cycle
      (pedest = 0); // no pedestrian
      (old_time = current_time); // reset the time
      Serial.println(pedest);
    }
  }
  
}

if (track == 1) { // this is the cycle with a pedestrian
  if (current_time - old_time < 4000) { // start with the yellow cycle
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, HIGH);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
  }
  if (current_time - old_time > 4000 && current_time - old_time < 19000) { //red cycle for 15 seconds for pedestrian
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(PED_RED, LOW);
  digitalWrite(PED_GREEN, HIGH); // green light for pedestrian
  }
  if (current_time - old_time > 19000 && current_time - old_time < 31000) {// green cycle for 12 seconds
  (greenlight = 1);
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(PED_RED, HIGH);
  digitalWrite(PED_GREEN, LOW);
  }
  if (current_time - old_time > 31000) {
    if (pedest == 1) {
      (greenlight = 0);
      (track = 1);
      (pedest = 0);
      (old_time = current_time);
      Serial.println(pedest);
    } else {
      (greenlight = 0);
      (track = 0);
      (pedest = 0);
      (old_time = current_time);
      Serial.println(pedest);
    }
  }
 }
}
