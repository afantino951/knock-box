import requests

# Replace with your ESP32's IP address
esp32_ip = "192.168.137.185"

# Define the servo angle you want to set
servo_angle = 90  # Change this to the desired angle

# Create the URL for the HTTP request
url = f"http://{esp32_ip}/?value={servo_angle}"

try:
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully set servo angle to {servo_angle} degrees.")
    else:
        print(f"Failed to set servo angle. Status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")