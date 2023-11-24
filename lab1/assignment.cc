#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>

#define PA 2
#define PB 4
#define PC 1
#define PD 16
#define PE 15
#define PF 7
#define PG 9
#define PDP 0

// for anode display
char nums[10] = {0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x98};

// WPi pin numbers
char pins[8] = {PA, PB, PC, PD, PE, PF, PG, PDP};

void clear_pin ()
{
  int i;
  for (i = 0; i < 8; i++)
    digitalWrite(pins[i], 1);
}

void set_pin (int n)
{
  int i;
  for (i = 0; i < 8; i++)
    digitalWrite(pins[i], (nums[n] >> i) & 0x1);
}

void init_pin()
{
  int i;
  for(i = 0; i < 8; i++)
    pinMode(pins[i], OUTPUT);
}


using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }



int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  wiringPiSetup();
  init_pin();

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);


  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver); 
  std::unique_ptr<tflite::Interpreter> interpreter; 
  builder(&interpreter); 
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);


  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk); 

  cv::VideoCapture video(0);

  cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
  if (!video.isOpened())
  {
    std::cout << "Unable to get video from the camera!" << std::endl;
    return -1;
  }  

  cv::Mat frame;

  vector<vector<float>> input_vector(28, vector<float>(28, 0));

  int frame_num = 0;


  while (video.read(frame))
  { 

    // // Display the frame
    cv::imshow("Camera Feed", frame);

    ++frame_num; 
    if (frame.empty()) {
          std::cerr << "Error: Empty frame." << std::endl;
          break;
      }

    if (frame_num % 100 == 0) {
      cv::resize(frame, frame, cv::Size(28, 28));
      cv::imshow("Camera Feed", frame);
      for (int i = 0; i < frame.rows; ++i) {
        for (int j = 0; j < frame.cols; ++j) {
            input_vector[i][j] = static_cast<int>(frame.at<cv::Vec3b>(i, j)[0]);
        }
      }

      std::cout << "Input MNIST Image" << "\n";
      for(int i=0; i<28; ++i){
        for(int j=0; j<28; ++j){
          printf("%3d ", (int)input_vector[i][j]);
        }
        printf("\n");
      }

      auto input_tensor = interpreter->typed_input_tensor<float>(0); 
      for(int i=0; i<28; ++i) 
        for(int j=0; j<28; ++j) 
          input_tensor[i * 28 + j] = input_vector[i][j] / 255.0; 
      

      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk); 
      printf("\n\n=== Post-invoke Interpreter State ===\n");
  
      auto output_tensor = interpreter->typed_output_tensor<float>(0); 
      for(int i=0; i<10; ++i)
        printf("label : %d %.3f% \n", i, output_tensor[i] * 100);

      int predict_num = 0;
      float max_value = output_tensor[0];

      // Iterate over the elements to find the maximum value and its index
      for (int i = 1; i < 10; ++i) {
        if (output_tensor[i] > max_value) {
            max_value = output_tensor[i];
            predict_num = i;
        }
      } 

      clear_pin();
      set_pin(predict_num);

    }  

    if (cv::waitKey(25) >= 0)
        break;


  }

  cv::destroyAllWindows();
  video.release();  
  return 0;
}


