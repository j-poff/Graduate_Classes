const int microphonePin = 0;
const int red_light_pin = 11;
const int green_light_pin = 10;
const int  blue_light_pin = 9;
//use a 10k resistor!!!

#define AVG_SIZE 3
float avgReadings[AVG_SIZE];
byte avgReadingSpot = 0;

void setup() {
  // put your setup code here, to run once:
pinMode(red_light_pin,OUTPUT);
pinMode(green_light_pin,OUTPUT);
pinMode(blue_light_pin,OUTPUT);
Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  int sound;
  sound = analogRead(microphonePin);
  //Serial.println(sound);

  avgReadings[avgReadingSpot] = sound;
  if(avgReadingSpot == AVG_SIZE){
      avgReadingSpot = 0;
    

    float avgReading = 0;
    for (int x = 0 ; x < AVG_SIZE ; x++)
      avgReading += avgReadings[x];
    avgReading /= AVG_SIZE;

    //Serial.print("AvgReading: ");
    //Serial.println(avgReading, 2); //Print 2 decimal places

    if ( abs(722-avgReading) >= 2 && abs(722-avgReading) < 4) {
    RGB_color(255,0,0);
    Serial.println("This is Hillside");
    delay(5000);
  } else if ( abs(722-avgReading) > 4) { 
    RGB_color(0,0,255);
    Serial.println("This is Radar");
    delay(5000);
  } else {
    RGB_color(0,0,0);
  }

    }

  avgReadingSpot = avgReadingSpot + 1;
  
  
  
}

void RGB_color(int red_light_value, int green_light_value, int blue_light_value)
{
  analogWrite(red_light_pin, red_light_value);
  analogWrite(green_light_pin, green_light_value);
  analogWrite(blue_light_pin, blue_light_value);
}


