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




  

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver); // Build InterpreterBuilder with model and OpResolver
  std::unique_ptr<tflite::Interpreter> interpreter; 
  builder(&interpreter);  // Create Interpreter with InterpreterBuilder
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk); // Allocate Tensors(give actual memory buffer to tensors)
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get()); // Debug print


  cv::VideoCapture video(0);



  cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
  if (!video.isOpened())
  {
    std::cout << "Unable to get video from the camera!" << std::endl;
    return -1;
  }  

  cv::Mat frame;

  vector<vector<int>> input_vector(28, vector<int>(28, 0));

  int frame_num = 0;


  while (video.read(frame))
  { 
    cv::resize(frame, frame, cv::Size(28, 28));
    for (int i = 0; i < frame.rows; ++i) {
      for (int j = 0; j < frame.cols; ++j) {
          input_vector[i][j] = frame.at<int>(i, j);
      }
    }

    std::cout << "Input MNIST Image" << "\n";
    for(int i=0; i<28; ++i){
      for(int j=0; j<28; ++j){
        printf("%3d ", (int)input_vector[i][j]);
      }
      printf("\n");
    }

    
        // Print the raw pixel values
    // for (int i = 0; i < frame.rows; ++i) {
    //     for (int j = 0; j < frame.cols; ++j) {
    //         std::cout << static_cast<int>(frame.at<cv::Vec3b>(i, j)[0]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // cv::imshow("Video feed", frame);

    // for (int i = 0; i < frame.rows; ++i) {
    //     for (int j = 0; j < frame.cols; ++j) {
    //         input_vector[i][j] = frame.at<int>(i, j);
    //     }
    // }
  // Convert the frame to grayscale
    //cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // Resize the frame to 28x28
    //cv::resize(frame, frame, cv::Size(28, 28));
//    cv::normalize(frame, frame, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // double minVal, maxVal;
    // cv::minMaxLoc(frame, &minVal, &maxVal);

    // for (int i = 0; i < frame.rows; ++i) {
    //     for (int j = 0; j < frame.cols; ++j) {
    //         frame.at<uchar>(i, j) = static_cast<uchar>(255.0 * (frame.at<uchar>(i, j) - minVal) / (maxVal - minVal));
    //     }
    // }



    // // Display the frame
    // cv::imshow("Camera Feed", frame);

    ++frame_num; 
    if (frame.empty()) {
          std::cerr << "Error: Empty frame." << std::endl;
          break;
      }

    if (frame_num % 100 == 0) {














      // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
      
      cv::resize(frame, frame, cv::Size(28, 28));

      // //cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
      cv::normalize(frame, frame, 0, 255, cv::NORM_MINMAX);
      // // // Convert the frame to grayscale (if not already)
      // // if (frame.channels() > 1) {
      // //     cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
      // // }

    


      // for (int i = 0; i < frame.rows; ++i) {
      //     for (int j = 0; j < frame.cols; ++j) {
      //         input_vector[i][j] = frame.at<int>(i, j);
      //     }
      // }

      // std::cout << "Input MNIST Image" << "\n";
      // for(int i=0; i<28; ++i){
      //   for(int j=0; j<28; ++j){
      //     printf("%3d ", (int)input_vector[i][j]);
      //   }
      //   printf("\n");
      // }


      // Fill input buffers
      // TODO(user): Insert code to fill input tensors.
      // Note: The buffer of the input tensor with index `i` of type T can
      // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
      auto input_tensor = interpreter->typed_input_tensor<float>(0); // Get input tensor's data field
      for(int i=0; i<28; ++i) // image rows
        for(int j=0; j<28; ++j) // image cols
          input_tensor[i * 28 + j] = input_vector[i][j] / 255.0; // normalize and copy input values.
      

      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk); // Do inference
      printf("\n\n=== Post-invoke Interpreter State ===\n");
      //tflite::PrintInterpreterState(interpreter.get());

      // Read output buffers
      // TODO(user): Insert getting data out code.
      // Note: The buffer of the output tensor with index `i` of type T can
      // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
      auto output_tensor = interpreter->typed_output_tensor<float>(0); // Get output tensor's data field
      for(int i=0; i<10; ++i)
        printf("label : %d %.3f% \n", i, output_tensor[i] * 100);

      // int predict_num = std::distance(output_tensor.begin(), std::max_element(output_tensor.begin(), output_tensor.end()));


      int predict_num = 0;
      float max_value = output_tensor[0];

      // Iterate over the elements to find the maximum value and its index
      for (int i = 1; i < 10; ++i) {
        if (output_tensor[i] > max_value) {
            max_value = output_tensor[i];
            predict_num = i;
        }
      } 


      //int k = 5;

      clear_pin();
      set_pin(predict_num);
    }  
    if (cv::waitKey(25) >= 0)
        break;


  }

  cv::destroyAllWindows();
  /*used to close all OpenCV windows created during the program's execution.*/
  video.release();
  /*This line releases the camera resource held by the video object.*/






  
  return 0;
}



// #include <wiringPi.h>
// #include <stdio.h>
// #include <stdlib.h>

// #define PA 2
// #define PB 4
// #define PC 1
// #define PD 16
// #define PE 15
// #define PF 8
// #define PG 9
// #define PDP 0

// // for anode display
// char nums[10] = {0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90};

// // WPi pin numbers
// char pins[8] = {PA, PB, PC, PD, PF, PG, PDP};

// void clear_pin ()
// {
//   int i;
//   for (i = 0; i < 8; i++)
//     digitalWrite(pins[i], 1);
// }

// void set_pin (int n)
// {
//   int i;
//   for (i = 0; i < 8; i++)
//     digitalWrite(pins[i], (nums[n] >> i) & 0x1);
// }

// void init_pin()
// {
//   int i;
//   for(i = 0; i < 8; i++)
//     pinMode(pins[i], OUTPUT);
// }

// int main(void)
// {
//   int i;
//   wiringPiSetup();
//   for (i=0; i < 10; i++)
//   {
//     clear_pin();
//     set_pin(i);
//     delay(1000);
//   }

//   return 0;
// }