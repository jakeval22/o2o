#include <Servo.h>
#include <RunningAverage.h>

#include <SPI.h>
#include <mcp2515.h>
MCP2515 mcp2515(53); //set to correct CS Pin
struct can_frame canMsg;


#define encoder 2
#define red_button 3
#define INITIAL 1499
#define ERROR_BOUND 20

RunningAverage avgRPM(30); 
Servo ESC;
Servo Steering;

int samples =0;
float desired_RPM;

// RPM globals
unsigned int rpm = 0;
volatile byte pulses = 0;
unsigned long TIME = 0;
unsigned volatile int ms = 1499;
unsigned int pulse_per_turn = 20; 
//depends on the number of slots on the slotted disc

// Control Global
int deviation = 0;
int error;


void setup() {
  // Configure Throttle
  ESC.attach(9,0,2000);       // (pin, min pulse width, max pulse width in us)
  Steering.attach(10,0,2000);   // (pin, min pulse width, max pulse width in us)
  
  Serial.begin(9600);

  // Initialization of Throttle
  ESC.writeMicroseconds(INITIAL);
  Steering.writeMicroseconds(INITIAL);
  
  // Configure Control Buttons
  pinMode(red_button, INPUT);
  attachInterrupt(digitalPinToInterrupt(red_button), redButton,FALLING);
  
  // Configure Encoder and Interrupts
  pinMode(encoder, INPUT);// setting up encoder pin as input
  attachInterrupt(digitalPinToInterrupt(encoder), count, FALLING);  //triggering count function everytime the encoder turns from HIGH to LOW
  
  desired_RPM = 0;

  avgRPM.clear(); // explicitly clean buffer
  delay(3000);

  //Serial.begin(115200);
  Serial.println("Setup Started!");
  //initialize MCP2515
  mcp2515.reset();
  mcp2515.setBitrate(CAN_125KBPS);
  mcp2515.setNormalMode();
  Serial.println("Setup done!");
  delay(1000);
  
  ms = 1560;
}

int counter=0;
int timeout=0;
int current_RPM=0;
float average= 0;
int steer_angle= 30;
int steer_input;


void loop() {
  // put your main code here, to run repeatedly:
  if (mcp2515.readMessage(&canMsg) == MCP2515::ERROR_OK) {
    Serial.print("New Message! \nFrom: "); // Declares readiness
    Serial.print(canMsg.can_id, HEX); // print ID
    Serial.print("\nMessage length: "); 
    Serial.print(canMsg.can_dlc, HEX); // print DLC
    Serial.print("\nMessage: ");
    if (canMsg.can_dlc == 2) {
      desired_RPM = canMsg.data[0];
      steer_angle = canMsg.data[1];
      Serial.println(desired_RPM);
      steer_input = map(steer_angle,0,60,15,165);
      steeringControl(steer_input);
    } 
  }
  
  //handleSerial();
  current_RPM = readEncoder();
  Serial.println(current_RPM);
  avgRPM.addValue(current_RPM);
  timeout = millis();
  
//  if( timeout >= 14000){
//    while(1){
//      ESC.writeMicroseconds(INITIAL);
//      //Wheels.writeMicroseconds(INITIAL);
//      //Serial.println(" DONE ");
//    }
//  }
  speedControl();

  
}

void steeringControl(int steer_angle){
  
  Steering.write(steer_angle);
  //Serial.println(steer_angle);
}

void speedControl(){
    // Every X amount of cycles, calculate average RPM and adjust PWM signal
  if (counter >= 45){
    average = avgRPM.getAverage();
    deviation = desired_RPM - average;
    error = abs(deviation);

    // Do not change PWM output if error too small
    if ((error <= (ERROR_BOUND) )){
      deviation = 0;
      error = 0;
    }
    // Otherwise, map deviation to throttle PWM output linearly
    else{

      if (deviation <= -7){
        ms = ms - 1;
      }
      if (deviation >= 7){
        ms = ms + 1;
      }

    }
    // Reset counter
    counter = 0;
  }

  // Max PWM ouput
  if(ms >= 1750){
    ms = 1750;
  }

  
  ESC.writeMicroseconds(ms);

  //double spd = rpm *0.2899 /60;
  
  counter+=1;
}


// Reads amount of pulses encountered and calculates RPM
int readEncoder(){
    if (millis() - TIME >= 200){ // updating every 0.2 second
    detachInterrupt(digitalPinToInterrupt(encoder)); // disable interrupt trigger
    //calcuate rpm 
    rpm = (60 *100 / pulse_per_turn)/ (millis() - TIME) * pulses;
    TIME = millis();
    pulses = 0; // reset pulses
    
    //trigger count function everytime the encoder turns from HIGH to LOW
    attachInterrupt(digitalPinToInterrupt(encoder), count, FALLING);
  }
  return rpm;
}

// Encoder Pulse Interrupt Service Routine
void count(){
  // counting the number of pulses for calculation of rpm
  pulses++;
}

// Emergency Stop Button Interrupt Service Routine
void redButton(){
  noInterrupts();
  ms = 1499;
  while(1){
    ESC.writeMicroseconds(ms);
    Steering.write(30);
    Serial.println("HALTED"); 
  }
}
/// Loop here
