/*
 * ESP32-S3 + INMP441 — fully on-device "Vaani" keyword spotting
 *
 * Pipeline (no server / no WiFi needed):
 *   I2S mic -> rolling 1s ring buffer -> energy VAD -> log-mel
 *   features -> TFLite Micro inference -> LED on/off
 *
 * Mirrors the VAD + feature-extraction logic of test_custom_audio_vad.py,
 * with ONE deliberate approximation: librosa used n_fft=480 (not a
 * power of 2). ESP-DSP's FFT needs a power of 2, so each 480-sample
 * Hann-windowed frame is zero-padded to 512 before transforming, and
 * mel_filterbank.h was generated to match n_fft=512 (not 480) so the
 * filters line up with the FFT bins actually produced here. This is
 * a close approximation of the python pipeline, not a bit-exact
 * match — see the note at the end of the chat reply on validating it.
 *
 * Library dependency: ESP-DSP (espressif/esp-dsp) for the FFT.
 *   PlatformIO: add `espressif/esp-dsp` to lib_deps.
 *   Arduino IDE: install "ESP-DSP" via Library Manager.
 *
 * Needs model_data.h (your existing exported int8 TFLite model) in
 * the same folder — not included here, reuse the one from your
 * TFLite Micro test sketch.
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <string.h>

#include "esp_dsp.h"

#include "model_data.h"
#include "mel_filterbank.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================
// Pins
// ============================================================
#define I2S_SD 33
#define I2S_WS 25
#define I2S_SCK 26
#define I2S_PORT I2S_NUM_0

#define LED_PIN 2 // onboard LED on most ESP32 dev boards — change if yours is elsewhere

// ============================================================
// Audio / feature config — mirrors test_custom_audio_vad.py
// ============================================================
#define SAMPLE_RATE 16000
#define WINDOW_SAMPLES 16000 // 1 second

#define HOP_MS 200          // run KWS every 200 ms, like HOP_SECONDS in python
#define MEL_HOP_SAMPLES 320 // 20 ms
#define NUM_FRAMES 49       // 1 + (WINDOW_SAMPLES - KWS_WIN_LEN) / MEL_HOP_SAMPLES

#define VAD_FRAME_SAMPLES 480 // 30 ms
#define VAD_THRESHOLD_DB -45.0f
#define VAD_START_FRAMES 2
#define VAD_END_FRAMES 5

#define KWS_THRESHOLD 0.90f

// ============================================================
// Ring buffer (producer: mic task, consumer: KWS task)
// ============================================================
static int16_t *ring;
static volatile size_t ring_write_pos = 0;
static SemaphoreHandle_t ring_mutex;

#define I2S_READ_CHUNK 1024
static int32_t *i2s_raw_buf;

// ============================================================
// TFLite Micro globals
// ============================================================
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

constexpr size_t kTensorArenaSize = 64 * 1024;
static uint8_t *tensor_arena;

// ============================================================
// Static working buffers (kept off the stack)
// ============================================================
static int16_t *window_audio;
static float *fft_buf;
static float *mel_power;
static float *features;

bool allocate_buffers()
{
  ring = static_cast<int16_t *>(heap_caps_calloc(
      WINDOW_SAMPLES, sizeof(*ring), MALLOC_CAP_8BIT));
  i2s_raw_buf = static_cast<int32_t *>(heap_caps_malloc(
      I2S_READ_CHUNK * sizeof(*i2s_raw_buf), MALLOC_CAP_8BIT));
  window_audio = static_cast<int16_t *>(heap_caps_malloc(
      WINDOW_SAMPLES * sizeof(*window_audio), MALLOC_CAP_8BIT));
  fft_buf = static_cast<float *>(heap_caps_malloc(
      2 * KWS_N_FFT * sizeof(*fft_buf), MALLOC_CAP_8BIT));
  mel_power = static_cast<float *>(heap_caps_malloc(
      KWS_N_MELS * NUM_FRAMES * sizeof(*mel_power), MALLOC_CAP_8BIT));
  features = static_cast<float *>(heap_caps_malloc(
      KWS_N_MELS * NUM_FRAMES * sizeof(*features), MALLOC_CAP_8BIT));

  uint8_t *arena_storage = static_cast<uint8_t *>(heap_caps_malloc(
      kTensorArenaSize + 15, MALLOC_CAP_8BIT));
  if (arena_storage != nullptr)
  {
    tensor_arena = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(arena_storage) + 15) & ~uintptr_t(15));
  }

  return ring != nullptr && i2s_raw_buf != nullptr &&
         window_audio != nullptr && fft_buf != nullptr &&
         mel_power != nullptr && features != nullptr && tensor_arena != nullptr;
}

// ============================================================
// I2S setup — same fix as the streaming firmware
// ============================================================
void i2s_install()
{
  const i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin()
{
  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD};
  i2s_set_pin(I2S_PORT, &pin_config);
}

// ============================================================
// Mic task: continuously fills the ring buffer
// ============================================================
void micTask(void *parameter)
{
  i2s_install();
  i2s_setpin();
  i2s_start(I2S_PORT);
  Serial.println("I2S microphone started");

  size_t bytesIn = 0;

  while (true)
  {
    esp_err_t result = i2s_read(
        I2S_PORT, i2s_raw_buf, I2S_READ_CHUNK * sizeof(int32_t),
        &bytesIn, portMAX_DELAY);

    if (result != ESP_OK || bytesIn == 0)
      continue;

    size_t samples_read = bytesIn / sizeof(int32_t);

    xSemaphoreTake(ring_mutex, portMAX_DELAY);
    for (size_t i = 0; i < samples_read; i++)
    {
      int32_t s = i2s_raw_buf[i] >> 16; // 24-bit -> 16-bit, same fix as the streaming firmware
      if (s > 32767)
        s = 32767;
      if (s < -32768)
        s = -32768;
      ring[ring_write_pos] = (int16_t)s;
      ring_write_pos = (ring_write_pos + 1) % WINDOW_SAMPLES;
    }
    xSemaphoreGive(ring_mutex);
  }
}

// ============================================================
// Snapshot the last WINDOW_SAMPLES into a linear, time-ordered buffer
// ============================================================
void snapshot_window(int16_t *out)
{
  xSemaphoreTake(ring_mutex, portMAX_DELAY);
  size_t pos = ring_write_pos; // oldest sample lives here
  size_t first_chunk = WINDOW_SAMPLES - pos;
  memcpy(out, &ring[pos], first_chunk * sizeof(int16_t));
  memcpy(out + first_chunk, &ring[0], pos * sizeof(int16_t));
  xSemaphoreGive(ring_mutex);
}

// ============================================================
// VAD — energy-based, ports vad_detect() from the python script
// ============================================================
float rms_dbfs(const float *frame, size_t n)
{
  if (n == 0)
    return -120.0f;
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; i++)
    sum_sq += (double)frame[i] * frame[i];
  double rms = sqrt(sum_sq / n);
  if (rms <= 1e-10)
    return -120.0f;
  return 20.0f * log10f((float)rms);
}

bool vad_detect(const int16_t *audio, size_t n)
{
  size_t num_frames = n / VAD_FRAME_SAMPLES;
  if (num_frames == 0)
    return false;

  int speech_frames = 0, silent_frames = 0;
  bool in_speech = false;
  static float frame_f[VAD_FRAME_SAMPLES];

  for (size_t i = 0; i < num_frames; i++)
  {
    for (size_t j = 0; j < VAD_FRAME_SAMPLES; j++)
    {
      frame_f[j] = audio[i * VAD_FRAME_SAMPLES + j] / 32768.0f;
    }
    float db = rms_dbfs(frame_f, VAD_FRAME_SAMPLES);
    bool is_speech = db > VAD_THRESHOLD_DB;

    if (is_speech)
    {
      speech_frames++;
      silent_frames = 0;
    }
    else
    {
      silent_frames++;
      speech_frames = 0;
    }

    if (!in_speech && speech_frames >= VAD_START_FRAMES)
      in_speech = true;
    if (in_speech && silent_frames >= VAD_END_FRAMES)
      in_speech = false;
  }

  return in_speech;
}

// ============================================================
// Feature extraction — log-mel spectrogram matching the V1 frontend
// ============================================================
void extract_features(const int16_t *audio)
{
  float max_power = 1e-10f;

  for (int t = 0; t < NUM_FRAMES; t++)
  {
    size_t start = t * MEL_HOP_SAMPLES;

    // Hann window + zero-pad KWS_WIN_LEN(480) -> KWS_N_FFT(512)
    for (int i = 0; i < KWS_N_FFT; i++)
    {
      float s = (i < KWS_WIN_LEN)
                    ? (audio[start + i] / 32768.0f) * hann_window[i]
                    : 0.0f;
      fft_buf[2 * i] = s;
      fft_buf[2 * i + 1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_buf, KWS_N_FFT);
    dsps_bit_rev2r_fc32(fft_buf, KWS_N_FFT);

    // power spectrum, bins 0..KWS_N_BINS-1
    float power[KWS_N_BINS];
    for (int k = 0; k < KWS_N_BINS; k++)
    {
      float re = fft_buf[2 * k];
      float im = fft_buf[2 * k + 1];
      power[k] = re * re + im * im;
    }

    // mel filterbank
    for (int m = 0; m < KWS_N_MELS; m++)
    {
      float acc = 0.0f;
      for (int k = 0; k < KWS_N_BINS; k++)
      {
        acc += mel_filterbank[m][k] * power[k];
      }
      mel_power[m * NUM_FRAMES + t] = acc;
      if (acc > max_power)
        max_power = acc;
    }
  }

  // power_to_db(ref=max) then (x+80)/80 normalize — matches the python script
  float ref_db = 10.0f * log10f(max_power);
  int idx = 0;
  for (int m = 0; m < KWS_N_MELS; m++)
  {
    for (int t = 0; t < NUM_FRAMES; t++)
    {
      float p = mel_power[m * NUM_FRAMES + t];
      if (p < 1e-10f)
        p = 1e-10f;
      float db = 10.0f * log10f(p) - ref_db;
      if (db < -80.0f)
        db = -80.0f;
      if (db > 0.0f)
        db = 0.0f;
      features[idx++] = (db + 80.0f) / 80.0f; // (mel, time) order -> (40,49,1) layout
    }
  }
}

// ============================================================
// Quantize float features into the int8 input tensor and run inference
// ============================================================
float run_inference()
{
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  for (int i = 0; i < KWS_N_MELS * NUM_FRAMES; i++)
  {
    int32_t q = (int32_t)lroundf(features[i] / scale) + zero_point;
    if (q < -128)
      q = -128;
    if (q > 127)
      q = 127;
    input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("ERROR: Invoke() failed");
    return 0.0f;
  }

  // Class order matches test_custom_audio_vad.py: 0 = negative, 1 = Vaani
  return (output->data.int8[1] - output->params.zero_point) * output->params.scale;
}

// ============================================================
// KWS task: every HOP_MS, snapshot -> VAD -> (maybe) infer -> LED
// ============================================================
void kwsTask(void *parameter)
{
  while (true)
  {
    snapshot_window(window_audio);

    bool speech = vad_detect(window_audio, WINDOW_SAMPLES);

    if (!speech)
    {
      digitalWrite(LED_PIN, LOW);
      Serial.println("VAD=NO SPEECH   KWS=SKIP");
    }
    else
    {
      uint32_t t0 = micros();
      extract_features(window_audio);
      float prob = run_inference();
      uint32_t elapsed_ms = (micros() - t0) / 1000;

      if (prob >= KWS_THRESHOLD)
      {
        digitalWrite(LED_PIN, HIGH);
        Serial.printf("VAD=SPEECH  Vaani=%.4f  *** DETECTED ***  (%lums)\n", prob, elapsed_ms);
      }
      else
      {
        digitalWrite(LED_PIN, LOW);
        Serial.printf("VAD=SPEECH  Vaani=%.4f  (%lums)\n", prob, elapsed_ms);
      }
    }

    // Inference takes longer than HOP_MS, so a fixed delay avoids starving IDLE0.
    vTaskDelay(pdMS_TO_TICKS(HOP_MS));
  }
}

// ============================================================
// Model / interpreter setup — same pattern as your TFLite Micro test
// ============================================================
bool setup_model()
{
  model = tflite::GetModel(g_vaani_model_data);
  if (model == nullptr)
  {
    Serial.println("ERROR: Could not load model");
    return false;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.printf("ERROR: Model schema %d != runtime schema %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk)
  {
    Serial.println("ERROR: failed to register an op");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors() failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input == nullptr || output == nullptr)
  {
    Serial.println("ERROR: Input/output tensor missing");
    return false;
  }

  Serial.printf("Input scale=%.6f zero_point=%d\n", input->params.scale, input->params.zero_point);
  Serial.printf("Input shape: %d x %d x %d x %d\n",
                input->dims->data[0], input->dims->data[1],
                input->dims->data[2], input->dims->data[3]);

  return true;
}

// ============================================================
// Setup / loop
// ============================================================
void setup()
{
  Serial.begin(115200);
  delay(1000);
  Serial.println("Vaani on-device KWS starting...");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  if (!allocate_buffers())
  {
    Serial.println("ERROR: Could not allocate audio/inference buffers");
    while (true)
      delay(1000);
  }

  ring_mutex = xSemaphoreCreateMutex();

  if (!setup_model())
  {
    Serial.println("Model setup failed — halting");
    while (true)
      delay(1000);
  }
  Serial.println("Model ready.");

  dsps_fft2r_init_fc32(NULL, KWS_N_FFT);

  xTaskCreatePinnedToCore(micTask, "micTask", 8000, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(kwsTask, "kwsTask", 16000, NULL, 1, NULL, 0);
}

void loop()
{
  delay(1000);
}