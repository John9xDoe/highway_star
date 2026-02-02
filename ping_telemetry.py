import truck_telemetry

class Telemetry:
    def __init__(self):
        truck_telemetry.init()
        self.convert_const_mps_to_mph = 2.236936

    def get_telemetry_keys(self, keyword=None):
        data = truck_telemetry.get_data().keys()

        if keyword is not None:
            data = [key for key in data if keyword in key]

        return data

    def ping_data(self, vis=True):
        data = truck_telemetry.get_data()
        speed, rpm, rpm_max = data.get("speed") * self.convert_const_mps_to_mph, data.get("engineRpm"), data.get("engineRpmMax")
        if data and vis:
            print(f"Speed: {speed} | RPM: {rpm} | RPM_MAX: {rpm_max}")

        return speed, rpm, rpm_max


if __name__ == "__main__":
    telemetry_pinger = Telemetry()
    print(telemetry_pinger.get_telemetry_keys())