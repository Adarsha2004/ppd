/*
 * MSCP-Mamba real-time thermal change detection -- ESP32-S3 sensor node
 * ---------------------------------------------------------------------
 * Board : ESP32-S3-N16R8 (ESP32-S3-DevKitC-1 style)
 * Sensor: 7Semi MLX90640 breakout (32x24 IR array, 110x75 deg FOV, I2C 0x33)
 *
 * Job of this firmware (the heavy CNN+Mamba model runs on the laptop):
 *   1. read full 32x24 temperature frames from the MLX90640 over I2C @ 400 kHz (auto-recovers from bus errors)
 *   2. stream them to the laptop as binary packets (USB serial, or WiFi TCP)
 *   3. receive the detection result back and drive the RGB LED / alert pin
 *
 * Packet (little-endian, 1554 bytes):
 *   magic  4  : A5 5A 4D 58                ("\xA5\x5AMX")
 *   seq    u32: frame counter
 *   t_ms   u32: millis() at capture
 *   ta     i16: sensor ambient temperature, centi-degC
 *   flags  u16: bit0 = WiFi transport
 *   temps  i16[768]: object temperatures, centi-degC, row-major 24 rows x 32 cols
 *   crc    u16: CRC-16/CCITT-FALSE over everything after magic (seq..temps)
 *
 * Host -> ESP32 commands (single bytes):
 *   'D' change detected (LED red, alert pin HIGH)
 *   'C' scene clear     (LED green, alert pin LOW)
 *   '1','2','4','8','6' set MLX refresh 1/2/4/8/16 Hz *per subpage*
 *                       (full frame rate = half of that; default '6' = 16 Hz -> ~8 fps)
 * No command for 2 s -> LED blue (host not running).
 *
 * Libraries: "Adafruit MLX90640" (+ Adafruit BusIO), installed via Library Manager.
 * Works with arduino-esp32 core 2.x and 3.x.
 */
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MLX90640.h>

// ----------------------------- configuration --------------------------------
#define I2C_SDA        8        // ESP32-S3 GPIO8  -> MLX90640 SDA
#define I2C_SCL        9        // ESP32-S3 GPIO9  -> MLX90640 SCL
#define I2C_FREQ       400000   // 400 kHz: reliable with jumper wires + breakout pull-ups
                                // (1 MHz works only with short wires / strong pull-ups)
#define ALERT_PIN      4        // optional LED / active buzzer (via resistor / transistor)
#define SERIAL_BAUD    921600   // ignored by native USB-CDC, used by the UART "COM" port
#define HOST_TIMEOUT_MS 2000

// WiFi transport (optional). Create wifi_secrets.h next to this file:
//   #define WIFI_SSID "..."
//   #define WIFI_PASS "..."
// If the file exists, frames go over TCP port 3333 (and USB serial still works).
#if __has_include("wifi_secrets.h")
  #include "wifi_secrets.h"
  #define USE_WIFI 1
#else
  #define USE_WIFI 0
#endif

#if USE_WIFI
  #include <WiFi.h>
  #include <ESPmDNS.h>
  #define TCP_PORT 3333
  WiFiServer server(TCP_PORT);
  WiFiClient client;
#endif

// Onboard WS2812 RGB LED: GPIO48 on DevKitC-1 v1.0 (most N16R8 boards), GPIO38 on v1.1.
#ifndef RGB_LED_PIN
  #ifdef RGB_BUILTIN
    #define RGB_LED_PIN RGB_BUILTIN
  #else
    #define RGB_LED_PIN 48
  #endif
#endif

// ----------------------------- globals --------------------------------------
Adafruit_MLX90640 mlx;
static float    frame[32 * 24];
static uint8_t  pkt[1554];
static uint32_t seq = 0;
static uint32_t lastCmdMs = 0;
static char     ledState = 'N';   // N=no host, D=detected, C=clear
static uint16_t errStreak = 0;
static mlx90640_refreshrate_t curRate = MLX90640_16_HZ;

static void setRgb(uint8_t r, uint8_t g, uint8_t b) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  rgbLedWrite(RGB_LED_PIN, r, g, b);
#else
  neopixelWrite(RGB_LED_PIN, r, g, b);
#endif
}

static void showState(char s) {
  if (s == ledState) return;
  ledState = s;
  switch (s) {
    case 'D': setRgb(40, 0, 0);  digitalWrite(ALERT_PIN, HIGH); break;   // red
    case 'C': setRgb(0, 25, 0);  digitalWrite(ALERT_PIN, LOW);  break;   // green
    default:  setRgb(0, 0, 25);  digitalWrite(ALERT_PIN, LOW);  break;   // blue
  }
}

// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) == Python binascii.crc_hqx(data, 0xFFFF)
static uint16_t crc16(const uint8_t *d, size_t n) {
  uint16_t crc = 0xFFFF;
  while (n--) {
    crc ^= (uint16_t)(*d++) << 8;
    for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
  }
  return crc;
}

static inline void put16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
static inline void put32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }

static void setRefresh(char c) {
  mlx90640_refreshrate_t r;
  switch (c) {
    case '1': r = MLX90640_1_HZ;  break;
    case '2': r = MLX90640_2_HZ;  break;
    case '4': r = MLX90640_4_HZ;  break;
    case '8': r = MLX90640_8_HZ;  break;
    case '6': r = MLX90640_16_HZ; break;
    case '3': r = MLX90640_32_HZ; break;   // noisier; needs 1 MHz I2C
    default: return;
  }
  curRate = r;
  mlx.setRefreshRate(r);
}

static void handleByte(int c) {
  if (c == 'D' || c == 'C') { showState((char)c); lastCmdMs = millis(); }
  else if (c == 'P')        { lastCmdMs = millis(); }
  else                      { setRefresh((char)c); }
}

static void pollCommands() {
  while (Serial.available()) handleByte(Serial.read());
#if USE_WIFI
  if (client && client.connected()) {
    while (client.available()) handleByte(client.read());
  }
#endif
  if (millis() - lastCmdMs > HOST_TIMEOUT_MS) showState('N');
}

static void sendPacket(const uint8_t *buf, size_t n) {
  // USB-CDC skips writes while no host has the port open; UART just drains at 921600.
  // A partially-sent packet is harmless: the host resyncs on magic + CRC.
  Serial.write(buf, n);
#if USE_WIFI
  if (server.hasClient()) {                       // newest client wins
    if (client) client.stop();
    client = server.accept();
    client.setNoDelay(true);
  }
  if (client && client.connected()) client.write(buf, n);
#endif
}

// ---------------------------------------------------------------------------
// Wiring auto-detect. I2C lines are pulled up to 3V3 by the sensor board, so a GPIO
// that still reads HIGH with the ESP32's internal pull-DOWN enabled is wired to SDA
// or SCL. Try every such pair (both orders) and keep the one where the MLX answers.
// Excluded: 0/3/45/46 strapping, 19/20 USB, 26-37 flash/PSRAM, 43/44 UART0, 48 LED.
// ---------------------------------------------------------------------------
static int sdaPin = I2C_SDA, sclPin = I2C_SCL;
static const int CAND[] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           21, 38, 39, 40, 41, 42, 47};

static bool tryPins(int sda, int scl) {
  Wire.end();
  Wire.begin(sda, scl);
  Wire.setClock(100000);
  Wire.beginTransmission(MLX90640_I2CADDR_DEFAULT);
  return Wire.endTransmission() == 0;
}

static bool autoDetectPins() {
  Wire.end();
  int up[8], nUp = 0;
  String list;
  for (int p : CAND) {
    if (p == RGB_LED_PIN || p == ALERT_PIN) continue;
    pinMode(p, INPUT_PULLDOWN);
    delayMicroseconds(200);
    if (digitalRead(p) == HIGH && nUp < 8) { up[nUp++] = p; list += " GPIO" + String(p); }
    pinMode(p, INPUT);
  }
  Serial.printf("# pins pulled up by the sensor:%s\n", nUp ? list.c_str() : " NONE");
  if (nUp < 2)
    Serial.println("# -> need 2 (SDA + SCL). A data wire is not connected to any ESP32 GPIO"
                   " (wrong breadboard row, or on GND / 3V3 / 5V).");
  for (int a = 0; a < nUp; a++)
    for (int b = 0; b < nUp; b++)
      if (a != b && tryPins(up[a], up[b])) {
        sdaPin = up[a]; sclPin = up[b];
        Serial.printf("# MLX90640 found: SDA=GPIO%d SCL=GPIO%d\n", sdaPin, sclPin);
        return true;
      }
  Wire.end();
  return false;
}

static bool initSensor() {
  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) return false;
  mlx.setMode(MLX90640_CHESS);
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(curRate);
  return true;
}

static void recoverSensor() {
  Serial.println("# re-initialising I2C + MLX90640");
  setRgb(40, 0, 40);                                // purple while recovering
  Wire.end();
  delay(20);
  Wire.begin(sdaPin, sclPin);
  Wire.setClock(I2C_FREQ);
  bool ok = initSensor();
  Serial.println(ok ? "# sensor re-initialised" : "# sensor not responding - check wiring");
  char s = ledState; ledState = 0; showState(s);    // restore LED
}

void setup() {
  Serial.setTxBufferSize(4096);                   // >= 2 packets; must precede begin()
  Serial.begin(SERIAL_BAUD);
  pinMode(ALERT_PIN, OUTPUT);
  digitalWrite(ALERT_PIN, LOW);
  ledState = 0; showState('N');

  Wire.setBufferSize(1024);                       // must precede Wire.begin() on ESP32
  delay(1500);                                    // give the USB host time to open the port
  for (;;) {
    if (tryPins(I2C_SDA, I2C_SCL) || autoDetectPins()) {
      Wire.end();
      Wire.begin(sdaPin, sclPin);
      Wire.setClock(I2C_FREQ);
      if (initSensor()) break;
    }
    Serial.printf("# MLX90640 not found (expected SDA=GPIO%d SCL=GPIO%d, 3V3, GND) - retrying\n",
                  I2C_SDA, I2C_SCL);
    setRgb(40, 0, 40); delay(500); setRgb(0, 0, 0); delay(500);   // purple blink = sensor error
  }
  Serial.printf("# MLX90640 ready on SDA=GPIO%d SCL=GPIO%d\n", sdaPin, sclPin);             // 16 Hz subpages -> ~8 full frames/s

#if USE_WIFI
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                           // lower latency
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) delay(200);
  if (WiFi.status() == WL_CONNECTED) {
    MDNS.begin("thermal-cam");                    // -> thermal-cam.local
    server.begin();
    Serial.printf("# WiFi %s  ->  tcp://%s:%d  (thermal-cam.local)\n",
                  WIFI_SSID, WiFi.localIP().toString().c_str(), TCP_PORT);
  } else {
    Serial.println("# WiFi connect failed - USB serial only");
  }
#endif
}

void loop() {
  pollCommands();
  if (mlx.getFrame(frame) != 0) {                  // blocks until both subpages are read
    errStreak++;
    if (errStreak == 1 || errStreak % 20 == 0) Serial.printf("# frame read error (x%u)\n", errStreak);
    if (errStreak % 10 == 0) recoverSensor();       // I2C glitch: reset the bus + sensor
    return;
  }
  if (errStreak) { Serial.printf("# sensor recovered after %u errors\n", errStreak); errStreak = 0; }
  uint8_t *p = pkt;
  p[0] = 0xA5; p[1] = 0x5A; p[2] = 'M'; p[3] = 'X'; p += 4;
  put32(p, seq++);  p += 4;
  put32(p, millis()); p += 4;
  put16(p, (uint16_t)(int16_t)lroundf(mlx.getTa(false) * 100.0f)); p += 2;
  put16(p, USE_WIFI ? 1 : 0); p += 2;
  for (int i = 0; i < 768; i++) {
    float t = frame[i];
    if (!isfinite(t)) t = -273.0f;
    t = constrain(t, -300.0f, 320.0f);
    put16(p, (uint16_t)(int16_t)lroundf(t * 100.0f)); p += 2;
  }
  put16(p, crc16(pkt + 4, (size_t)(p - (pkt + 4)))); p += 2;
  sendPacket(pkt, (size_t)(p - pkt));
}
