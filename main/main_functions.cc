/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "main_functions.h"
#include "model_int8.h"
#include "constants.h"
#include "output_handler.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 20000;
uint8_t tensor_arena[kTensorArenaSize];

// Define the classes
const char* kCategoryLabels[3] = {"cebollas", "limon", "papas"};
}  // namespace

// Function to load an image from the filesystem and preprocess it
bool LoadImage(const char* filename, TfLiteTensor* input_tensor) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        return false;
    }

    // For simplicity, let's assume buffer contains the preprocessed image data
    // This is a placeholder. Actual image processing code should go here.
    for (int i = 0; i < input_tensor->bytes; ++i) {
        input_tensor->data.int8[i] = static_cast<int8_t>(buffer[i]);
    }

    std::cout << "Successfully loaded image: " << filename << std::endl;
    return true;
}

// The name of this function is important for Arduino compatibility.
void setup() {
    tflite::InitializeTarget();

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(fruits);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model provided is schema version " << model->version()
                  << " not equal to supported version " << TFLITE_SCHEMA_VERSION << "." << std::endl;
        return;
    }

    // Add the required operations
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddFullyConnected();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        std::cerr << "AllocateTensors() failed" << std::endl;
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Keep track of how many inferences we have performed.
    inference_count = 0;

    
}

// The name of this function is important for Arduino compatibility.
void loop() {

  // Print image names to verify their presence
    const char* images[] = {"images/cebolla.jpg", "images/papa.jpg", "images/limon.jpg"};
    for (const char* image : images) {
        std::ifstream file(image, std::ios::binary);
        if (file) {
            std::cout << "Found image: " << image << std::endl;
        } else {
            std::cerr << "Image not found: " << image << std::endl;
        }
    }
    // Load a random image from the images folder
    srand(time(0));
    int random_index = rand() % 3;
    const char* image_path = images[random_index];

    if (!LoadImage(image_path, input)) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        std::cerr << "Invoke failed on image: " << image_path << std::endl;
        return;
    }

    // Obtain the quantized output from model's output tensor
    int max_index = 0;
    for (int i = 1; i < output->dims->data[1]; ++i) {
        if (output->data.int8[i] > output->data.int8[max_index]) {
            max_index = i;
        }
    }

    const char* predicted_label = kCategoryLabels[max_index];
    std::cout << "Image: " << image_path << ", Prediction: " << predicted_label << std::endl;

    // Increment the inference_counter, and reset it if we have reached the total number per cycle
    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
