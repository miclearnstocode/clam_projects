#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ===========================
// Camera Pin Definitions for AI-Thinker ESP32-CAM
// ===========================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ===========================
// WiFi Credentials
// ===========================
const char* ssid = "ZTE_2.4G_LSKYuG";
const char* password = "Connectka?_Pagbayadl4ng";

WebServer server(80);

// ===========================
// Web Server Handlers
// ===========================
void handleRoot() {
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<title>ESP32-CAM</title>";
  html += "<style>";
  html += "body{font-family:Arial;text-align:center;margin:20px;background:#1a1a1a;color:#fff;}";
  html += "h1{color:#4CAF50;}";
  html += "img{max-width:100%;height:auto;border:3px solid #4CAF50;border-radius:10px;}";
  html += "button{font-size:20px;padding:12px 30px;margin:10px;cursor:pointer;";
  html += "background:#4CAF50;color:#fff;border:none;border-radius:5px;}";
  html += "button:hover{background:#45a049;}";
  html += ".info{color:#aaa;font-size:14px;margin-top:20px;}";
  html += "</style></head><body>";
  html += "<h1>📷 ESP32-CAM</h1>";
  html += "<img id='camera' src='/capture' />";
  html += "<br><br>";
  html += "<button onclick='document.getElementById(\"camera\").src=\"/capture?t=\"+new Date().getTime()'>📸 Refresh</button>";
  html += "<div class='info'>Click Refresh to capture new image</div>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  
  // Send the image using the client directly
  WiFiClient client = server.client();
  
  // Build HTTP response headers
  String header = "HTTP/1.1 200 OK\r\n";
  header += "Content-Type: image/jpeg\r\n";
  header += "Content-Length: " + String(fb->len) + "\r\n";
  header += "Cache-Control: no-cache\r\n";
  header += "Connection: close\r\n\r\n";
  
  // Send headers
  client.print(header);
  
  // Send binary image data
  client.write(fb->buf, fb->len);
  
  // Close connection
  client.stop();
  
  esp_camera_fb_return(fb);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not Found");
}

// ===========================
// Setup
// ===========================
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n\n========================================");
  Serial.println("   ESP32-CAM Web Server Starting");
  Serial.println("========================================");

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;  // 320x240 - stable
  config.jpeg_quality = 15;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  Serial.println("Initializing camera...");
  esp_err_t err = esp_camera_init(&config);
  
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    Serial.println("\n=== TROUBLESHOOTING ===");
    Serial.println("1. Check ribbon cable - metal contacts face UP");
    Serial.println("2. Make sure cable is fully inserted");
    Serial.println("3. Try external 5V 2A power supply");
    return;
  }
  
  Serial.println("✅ Camera initialized successfully!");
  
  // Get sensor info
  sensor_t* s = esp_camera_sensor_get();
  Serial.printf("   Sensor: PID 0x%x\n", s->id.PID);
  if (s->id.PID == OV2640_PID) Serial.println("   Model: OV2640");
  else if (s->id.PID == OV3660_PID) Serial.println("   Model: OV3660");
  else if (s->id.PID == OV5640_PID) Serial.println("   Model: OV5640");

  // Connect to WiFi
  Serial.printf("\nConnecting to WiFi: %s\n", ssid);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✅ WiFi connected!");
    Serial.print("   IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("❌ WiFi connection failed!");
    return;
  }

  // Start web server
  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("✅ Web server started!");
  
  Serial.println("\n========================================");
  Serial.print("🌐 Open browser: http://");
  Serial.println(WiFi.localIP());
  Serial.println("========================================\n");
}

void loop() {
  server.handleClient();
  delay(1);
}