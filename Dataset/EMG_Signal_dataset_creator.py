import serial
import pandas as pd

port = 'COM5'  # change to your port
baud = 115200

ser = serial.Serial(port, baud)

data = []
label = input("Enter gesture label: ")

print("Collecting data... Press CTRL+C to stop")

try:
    while True:
        line = ser.readline().decode().strip()
        values = list(map(float, line.split(",")))

        if len(values) == 15:
            values.append(label)
            data.append(values)

except KeyboardInterrupt:
    print("Stopped")

columns = [f"f{i}" for i in range(15)] + ["label"]

df = pd.DataFrame(data, columns=columns)

file_name = "emg_dataset.csv"
df.to_csv(file_name, mode='a', index=False, header=False)

print(f"Saved {len(data)} samples")