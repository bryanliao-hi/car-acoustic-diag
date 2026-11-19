/*
 * car-acoustic-diag — ESP32-S3 I2S MEMS microphone capture prototype
 *
 * Reads an I2S MEMS microphone (e.g. INMP441 / SPH0645 class) via the
 * legacy ESP-IDF I2S driver (<driver/i2s.h>), converts samples to
 * int16 PCM, and packs them into a packed AudioFrame together with a
 * millisecond sync timestamp and the current vehicle speed.
 *
 * Frames are streamed over USB Serial as raw bytes:
 *   [0xAA 0x55] + AudioFrame (520 bytes, little-endian)
 * The host side re-syncs on the 2-byte header.
 *
 * Vehicle speed is a hook: readSpeedKmh() currently returns the last
 * value stored in g_speedKmh. Wire it to CAN / OBD-II / GPS in the
 * integration step; the frame format does not change.
 */

#include <Arduino.h>
#include <driver/i2s.h>

// ---------- Configuration ----------
#define I2S_PORT        I2S_NUM_0
#define I2S_BCK_PIN     5    // SCK / BCLK
#define I2S_WS_PIN      6    // LRCK / WS
#define I2S_DIN_PIN     7    // SD / DOUT from microphone
#define SAMPLE_RATE_HZ  16000
#define FRAME_SAMPLES   256

// ---------- Frame format (must match host-side parser) ----------
typedef struct __attribute__((packed)) {
  uint32_t sync_ms;                 // millis() at frame start
  float    speed_kmh;               // vehicle speed at frame start
  int16_t  pcm[FRAME_SAMPLES];      // mono PCM, 16 kHz
} AudioFrame;

static_assert(sizeof(AudioFrame) == 4 + 4 + FRAME_SAMPLES * 2,
              "AudioFrame layout changed — update the host parser");

// ---------- Vehicle speed hook ----------
static volatile float g_speedKmh = 0.0f;

static float readSpeedKmh() {
  // TODO(integration): replace with CAN / OBD-II PID 0x0D / GPS value.
  return g_speedKmh;
}

// ---------- I2S setup ----------
static bool setupI2s() {
  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE_HZ,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // MEMS mics output 24/32-bit
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = FRAME_SAMPLES,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0,
  };

  i2s_pin_config_t pins = {
    .bck_io_num = I2S_BCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_DIN_PIN,
  };

  if (i2s_driver_install(I2S_PORT, &config, 0, NULL) != ESP_OK) return false;
  if (i2s_pin_config(I2S_PORT, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_PORT);
  return true;
}

void setup() {
  Serial.begin(921600);
  while (!Serial) { delay(10); }
  if (!setupI2s()) {
    Serial.println("I2S init failed");
    while (true) { delay(1000); }
  }
}

void loop() {
  static int32_t raw[FRAME_SAMPLES];
  AudioFrame frame;
  size_t bytesRead = 0;

  frame.sync_ms = (uint32_t)millis();
  frame.speed_kmh = readSpeedKmh();

  esp_err_t err = i2s_read(I2S_PORT, raw, sizeof(raw), &bytesRead,
                           portMAX_DELAY);
  if (err != ESP_OK || bytesRead == 0) return;

  size_t n = bytesRead / sizeof(int32_t);
  if (n > FRAME_SAMPLES) n = FRAME_SAMPLES;
  for (size_t i = 0; i < n; i++) {
    // 32-bit I2S sample -> int16: keep the top 16 bits.
    frame.pcm[i] = (int16_t)(raw[i] >> 16);
  }
  for (size_t i = n; i < FRAME_SAMPLES; i++) frame.pcm[i] = 0;

  const uint8_t header[2] = {0xAA, 0x55};
  Serial.write(header, sizeof(header));
  Serial.write((const uint8_t *)&frame, sizeof(frame));
}
