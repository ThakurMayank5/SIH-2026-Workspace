/*
 * ESP32-S3 + INMP441
 * HTTP POST audio chunks
 */

#include <driver/i2s.h>
#include <WiFi.h>
#include <HTTPClient.h>

// -------------------- WiFi --------------------

const char *ssid = "ThakurMayank";
const char *password = "2c47G4=1";

// -------------------- Server --------------------

const char *serverURL = "http://10.104.200.44:8000/audio";

// -------------------- I2S --------------------

#define I2S_SD 33
#define I2S_WS 25
#define I2S_SCK 26
#define I2S_PORT I2S_NUM_0

// -------------------- Audio --------------------

#define SAMPLE_RATE 16000
#define bufferLen 1024

int32_t audioBuffer[bufferLen];

// -------------------- Function Prototypes --------------------

void connectWiFi();
void i2s_install();
void i2s_setpin();
void micTask(void *parameter);

// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("ESP32-S3 INMP441 HTTP Audio");

  connectWiFi();

  xTaskCreatePinnedToCore(
      micTask,
      "micTask",
      10000,
      NULL,
      1,
      NULL,
      1);
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
  delay(100);
}

// ============================================================
// WIFI
// ============================================================

void connectWiFi()
{
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");

  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

// ============================================================
// I2S CONFIGURATION
// ============================================================

void i2s_install()
{
  const i2s_config_t i2s_config =
      {
          .mode = (i2s_mode_t)(I2S_MODE_MASTER |
                               I2S_MODE_RX),

          .sample_rate = SAMPLE_RATE,

          .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

          .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,

          .communication_format = I2S_COMM_FORMAT_I2S,

          .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

          .dma_buf_count = 8,

          .dma_buf_len = 256,

          .use_apll = false,

          .tx_desc_auto_clear = false,

          .fixed_mclk = 0};

  // const i2s_config_t i2s_config = {
  //     .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
  //     .sample_rate = 44100,
  //     .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
  //     .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
  //     .communication_format = I2S_COMM_FORMAT_I2S,
  //     .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
  //     .dma_buf_count = 8,
  //     .dma_buf_len = 256,
  //     .use_apll = false,
  // };

  i2s_driver_install(
      I2S_PORT,
      &i2s_config,
      0,
      NULL);
}

// ============================================================
// I2S PINS
// ============================================================

void i2s_setpin()
{
  const i2s_pin_config_t pin_config =
      {
          .bck_io_num = I2S_SCK,
          .ws_io_num = I2S_WS,
          .data_out_num = I2S_PIN_NO_CHANGE,
          .data_in_num = I2S_SD};

  i2s_set_pin(
      I2S_PORT,
      &pin_config);
}

// ============================================================
// MICROPHONE TASK
// ============================================================

void micTask(void *parameter)
{
  // Initialize I2S
  i2s_install();
  i2s_setpin();

  i2s_start(I2S_PORT);

  Serial.println("I2S microphone started");
  Serial.println("Starting HTTP audio transmission...");

  size_t bytesIn = 0;

  while (true)
  {
    // Read audio from INMP441
    esp_err_t result = i2s_read(
        I2S_PORT,
        audioBuffer,
        bufferLen * sizeof(int32_t),
        &bytesIn,
        portMAX_DELAY);

    if (result == ESP_OK && bytesIn > 0)
    {
      // Check WiFi
      if (WiFi.status() != WL_CONNECTED)
      {
        Serial.println("WiFi disconnected!");
        continue;
      }

      // Create HTTP client
      HTTPClient http;

      http.begin(serverURL);

      // Raw binary audio
      http.addHeader(
          "Content-Type",
          "application/octet-stream");

      // Send audio chunk
      int httpResponseCode = http.POST(
          (uint8_t *)audioBuffer,
          bytesIn);

      Serial.print("Sent ");
      Serial.print(bytesIn);
      Serial.print(" bytes | HTTP response: ");
      Serial.println(httpResponseCode);

      http.end();
    }
  }
}