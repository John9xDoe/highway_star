import truck_telemetry
import time

truck_telemetry.init()
print("Check starting in 10 seconds")
time.sleep(10)

while True:
    data = truck_telemetry.get_data()
    print(data.get("coordinateX"), data.get("coordinateY"), data.get("coordinateZ"))
    time.sleep(3)

    # ['coordinateX', 'coordinateY', 'coordinateZ']