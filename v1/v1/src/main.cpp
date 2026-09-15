#include <Arduino.h>

#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ============================================================
// Model
// ============================================================

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// ============================================================
// Tensor arena
// ============================================================

// Start deliberately large.
// We will measure the actual requirement later.
constexpr size_t kTensorArenaSize = 64 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];


// ============================================================
// Setup
// ============================================================

void setup() {

    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("====================================");
    Serial.println("      VAANI TFLITE MICRO TEST");
    Serial.println("====================================");

    Serial.printf("Model size: %u bytes\n",
                  g_vaani_model_data_len);

    Serial.printf("Tensor arena: %u bytes\n",
                  kTensorArenaSize);


    // --------------------------------------------------------
    // Load model
    // --------------------------------------------------------

    model = tflite::GetModel(g_vaani_model_data);

    if (model == nullptr) {
        Serial.println("ERROR: Could not load model");
        return;
    }

    Serial.println("Model loaded");


    // Check model schema version
    if (model->version() != TFLITE_SCHEMA_VERSION) {

        Serial.printf(
            "ERROR: Model schema %d != runtime schema %d\n",
            model->version(),
            TFLITE_SCHEMA_VERSION
        );

        return;
    }

    Serial.println("Model schema OK");


    // --------------------------------------------------------
    // Register ONLY the operators our model uses
    // --------------------------------------------------------

    static tflite::MicroMutableOpResolver<6> resolver;

    if (resolver.AddConv2D() != kTfLiteOk) {
        Serial.println("ERROR: AddConv2D failed");
        return;
    }

    if (resolver.AddDepthwiseConv2D() != kTfLiteOk) {
        Serial.println("ERROR: AddDepthwiseConv2D failed");
        return;
    }

    if (resolver.AddMaxPool2D() != kTfLiteOk) {
        Serial.println("ERROR: AddMaxPool2D failed");
        return;
    }

    if (resolver.AddMean() != kTfLiteOk) {
        Serial.println("ERROR: AddMean failed");
        return;
    }

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        Serial.println("ERROR: AddFullyConnected failed");
        return;
    }

    if (resolver.AddSoftmax() != kTfLiteOk) {
        Serial.println("ERROR: AddSoftmax failed");
        return;
    }

    Serial.println("Operators registered");


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;


    // --------------------------------------------------------
    // Allocate tensors
    // --------------------------------------------------------

    TfLiteStatus status = interpreter->AllocateTensors();

    if (status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed");
        return;
    }

    Serial.println("Tensor allocation OK");


    // --------------------------------------------------------
    // Get input/output tensors
    // --------------------------------------------------------

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input == nullptr || output == nullptr) {
        Serial.println("ERROR: Input/output tensor missing");
        return;
    }


    // --------------------------------------------------------
    // Print tensor information
    // --------------------------------------------------------

    Serial.println();
    Serial.println("------------- INPUT -------------");

    Serial.printf(
        "Type: %d\n",
        input->type
    );

    Serial.printf(
        "Shape: %d x %d x %d x %d\n",
        input->dims->data[0],
        input->dims->data[1],
        input->dims->data[2],
        input->dims->data[3]
    );

    Serial.printf(
        "Scale: %.10f\n",
        input->params.scale
    );

    Serial.printf(
        "Zero point: %d\n",
        input->params.zero_point
    );


    Serial.println();
    Serial.println("------------- OUTPUT ------------");

    Serial.printf(
        "Type: %d\n",
        output->type
    );

    Serial.printf(
        "Scale: %.10f\n",
        output->params.scale
    );

    Serial.printf(
        "Zero point: %d\n",
        output->params.zero_point
    );


    // --------------------------------------------------------
    // Fill test input
    // --------------------------------------------------------

    // For now we use zero input.
    // This is NOT an actual audio test yet.

    const size_t input_elements =
        input->bytes;

    memset(input->data.int8, 0, input_elements);


    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    Serial.println();
    Serial.println("Running inference...");

    uint32_t start_us = micros();

    status = interpreter->Invoke();

    uint32_t elapsed_us = micros() - start_us;


    if (status != kTfLiteOk) {
        Serial.println("ERROR: Invoke() failed");
        return;
    }


    // --------------------------------------------------------
    // Print results
    // --------------------------------------------------------

    Serial.println("Inference OK");

    Serial.printf(
        "Inference time: %lu us\n",
        elapsed_us
    );

    Serial.printf(
        "Inference time: %.3f ms\n",
        elapsed_us / 1000.0f
    );


    // Output is INT8.
    // Dequantize:
    //
    // real_value =
    //      (quantized_value - zero_point) * scale
    //

    float output0 =
        (output->data.int8[0] - output->params.zero_point)
        * output->params.scale;

    float output1 =
        (output->data.int8[1] - output->params.zero_point)
        * output->params.scale;


    Serial.println();
    Serial.println("------------- OUTPUT ------------");

    Serial.printf(
        "Class 0: %.6f\n",
        output0
    );

    Serial.printf(
        "Class 1: %.6f\n",
        output1
    );


    Serial.println();
    Serial.println("====================================");
    Serial.println("       TFLITE TEST COMPLETE");
    Serial.println("====================================");
}


void loop() {

    delay(2000);
}