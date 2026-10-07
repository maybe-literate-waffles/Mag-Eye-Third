import serial
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- SERIAL CONFIGURATION ---
SERIAL_PORT = (
    "/dev/ttyUSB0"  # Linux/Mac: '/dev/ttyUSB0' or '/dev/ttyACM0' | Windows: 'COM3'
)
BAUD_RATE = 115200
NUM_SAMPLES = 91


def read_scan(ser):
    """Requests data from ESP32 and captures the array."""
    ser.write(b"S")
    distances = []

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "END":
            break
        if "," in line:
            try:
                angle, dist = line.split(",")
                distances.append(float(dist))
            except ValueError:
                continue

    if len(distances) < NUM_SAMPLES:
        distances.extend([400.0] * (NUM_SAMPLES - len(distances)))
    return np.array(distances[:NUM_SAMPLES])


# --- DUMMY DATA CREATION FOR TESTING ---
X_train = []
y_train = []

for _ in range(100):
    # Class 0: Continuous Flat Surface / Static Object (uniform distance profile)
    flat_dist = np.random.uniform(10, 150)
    flat_profile = np.full(NUM_SAMPLES, flat_dist) + np.random.normal(
        0, 0.5, NUM_SAMPLES
    )
    X_train.append(flat_profile)
    y_train.append("Static Target / Wall")

    # Class 1: Scanned Object (bell curve profile for future motor integration)
    bg_dist = 200.0
    cyl_profile = np.full(NUM_SAMPLES, bg_dist)
    obj_dist = np.random.uniform(20, 80)
    cyl_profile[35:56] = obj_dist + np.random.normal(0, 0.5, 21)
    X_train.append(cyl_profile)
    y_train.append("Localized Object")

X_train = np.array(X_train)
y_train = np.array(y_train)

# --- TRAIN MODEL ---
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("ML Model trained successfully.")


# --- INFERENCE EXECUTION ---
def run_pipeline():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Allow serial interface to stabilize

        print("Reading live single-point distance from ESP32...")
        sweep_data = read_scan(ser)

        # Calculate real measured distance
        current_distance = float(np.mean(sweep_data))

        # Predict class
        prediction = model.predict([sweep_data])[0]

        print("\n--- TEST DETECTION RESULT ---")
        print(f"Object Class : {prediction}")
        print(f"Distance     : {current_distance:.2f} cm")
        print("----------------------------")

        ser.close()
    except Exception as e:
        print(f"Serial Error: {e}")


if __name__ == "__main__":
    run_pipeline()
