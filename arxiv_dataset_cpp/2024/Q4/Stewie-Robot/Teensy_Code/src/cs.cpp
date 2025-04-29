// #include <Arduino.h>
// #include <Wire.h>
// #include <Servo.h>
// #include <MPU6050_tockn.h>
// #include<SD.h>


// File dataFile;


// #define SERVO_1_PIN 5 // Zpb3
// #define SERVO_2_PIN 4 // Zpb2
// #define SERVO_3_PIN 3 // Zpb1

// MPU6050 mpu6050(Wire);

// Servo servo1;
// Servo servo2;
// Servo servo3;

// float servo_lower_limit_deg = 0;
// float servo_upper_limit_deg = 90;
// float servo_range = servo_upper_limit_deg - servo_lower_limit_deg;

// float initial_servo_position = 0;


// float cumulative_error1 = 0;
// float previous_error1 = 0;

// float Kp1 = 1.5;
// float Ki1 = 0.05;
// float Kd1 = 0.1;


// float cumulative_error2 = 0;
// float previous_error2 = 0;

// float Kp2 = 1.5;
// float Ki2 = 0.05;
// float Kd2 = 0.1;


// float cumulative_error3 = 0;
// float previous_error3 = 0;

// float Kp3 = 1.5;
// float Ki3 = 0.05;
// float Kd3 = 0.1;



// float theta_x,theta_y, theta_z, Xpb1, Ypb1, Zpb1, Xpb2, Ypb2, Zpb2, Xpb3, Ypb3, Zpb3;
// float l12_1,l12_2,l12_3, a1, b1, c1, a2, b2,c2,a3,b3,c3,alpha1,alpha2,alpha3,l12_x1,l12_x2,l12_x3, l12_y1,l12_y2,l12_y3, l12_z1,l12_z2,l12_z3;
// float Xp1 = 0;
// float Xp2 = -39;
// float Xp3 = 38.9;
// float Yp1 = -41.8;
// float Yp2 = 21;
// float Yp3 = 22.5;
// float Zp1 = 0;
// float Zp2 = 0;
// float Zp3 = 0;
// float Ztb = 85.5; 
// float l1 = 15;
// float l2 = 85;
// float Mx1 = 35;
// float Mx2 = 12;
// float Mx3 = 12;

// float My1 = 0;
// float My2 = 30;
// float My3 = 30;

// float Mz1 = 0;
// float Mz2 = 0;
// float Mz3 = 0;

// float theta_1 = 0;
// float theta_2 = 120;
// float theta_3 = 240;


// // Servo angles (in degrees)
// float servoAngle1 = initial_servo_position;
// float servoAngle2 = initial_servo_position;
// float servoAngle3 = initial_servo_position;

// float z_coordinate (float theta_x, float theta_y, float theta_z, float Xp, float Yp, float Zp)

// {

//   float Zpb = Xp*(-1*sin(theta_y)) + Yp*(cos(theta_y)*sin(theta_x)) + Zp*(cos(theta_y)*cos(theta_x)) + Ztb;

//   return Zpb;
// }

// float middle(float a, float b, float c)
// {
//   // Checking for b
//     if ((a < b && b < c) || (c < b && b < a))
//        return b;
 
//     // Checking for a
//     else if ((b < a && a < c) || (c < a && a < b))
//        return a;
 
//     else
//        return c;
// }

// float pid(float kp, float ki, float kd, float *errorSum, float *lastError, float current_z, float desired_z)
// {
//     // input --> Z coordiate data from the calculations
//     // output --> New angle for servo to move to reach desired z coordiate  point

    
//     float error = desired_z - current_z;
//     float errorDiff = error - *lastError;

//     float pidOutput = kp * error + ki * *errorSum + kd * errorDiff;

//     *errorSum += error;
//     *lastError = error;

//     float newServoAngle = map(pidOutput, -1 * servo_range, servo_range, servo_lower_limit_deg, servo_upper_limit_deg);

//     if( newServoAngle > servo_upper_limit_deg )
//     {
//         newServoAngle = servo_upper_limit_deg;
//     }

//     if( newServoAngle < servo_lower_limit_deg)
//     {
//         newServoAngle = servo_lower_limit_deg;
//     }

//     return newServoAngle;

// }

// void setup() 
// {
//     Serial.begin(9600);
//     Wire.begin();
//     mpu6050.begin();
//     mpu6050.calcGyroOffsets(false);
//     servo1.attach(SERVO_1_PIN);
//     servo2.attach(SERVO_2_PIN);
//     servo3.attach(SERVO_3_PIN);
//     delay(50);

//     servo1.write(initial_servo_position);
//     servo2.write(initial_servo_position);
//     servo3.write(initial_servo_position);
// }

// void loop()
// {
//     theta_x = mpu6050.getAngleX() ;
//     theta_y = mpu6050.getAngleY();
//     theta_z = mpu6050.getAngleZ() ;

//     Zpb1 = z_coordinate(theta_x, theta_y, theta_z, Xp1, Yp1, Zp1);
//     Zpb2 = z_coordinate(theta_x, theta_y, theta_z, Xp2, Yp2, Zp2);
//     Zpb3 = z_coordinate(theta_x, theta_y, theta_z, Xp3, Yp3, Zp3);

//     float desired_angle = middle(Zpb1, Zpb2, Zpb3);

//     float servo_angle1 = pid(Kp1, Ki1, Kd1, &cumulative_error1, &previous_error1, Zpb1, desired_angle);
//     float servo_angle2 = pid(Kp2, Ki2, Kd2, &cumulative_error2, &previous_error2, Zpb2, desired_angle);
//     float servo_angle3 = pid(Kp3, Ki3, Kd3, &cumulative_error3, &previous_error3, Zpb3, desired_angle);
    

//     servo3.write(servo_angle1);
//     servo2.write(servo_angle2);
//     servo1.write(servo_angle3);

//     // Initialize SD card
//   if (!SD.begin(BUILTIN_SDCARD)) {
//     Serial.println("SD card initialization failed!");
//     return;
//   }
  
//   Serial.println("SD card initialized successfully.");
  
//   // Open file for writing
//   dataFile = SD.open("data.txt", FILE_WRITE);
  
//   if (!dataFile) {
//     Serial.println("Error opening data.txt");
//     return;
//   }
  
//   Serial.println("Writing data to data.txt...");
  
//   // Write some data to the file
//   String data = "";
//     data += String(theta_x) + ", " + String(theta_y) + ", " + String(theta_z) + ", " + String(Zpb1) + ", " + String(Zpb2) + ", " + String(Zpb3) + String(servo_angle1) + ", " + String(servo_angle2) + ", " + String(servo_angle3);
//     dataFile.println(data);
  
//   // Close the file
//   dataFile.close();
  
//   Serial.println("Data written to data.txt");

//     delay(50);


// }