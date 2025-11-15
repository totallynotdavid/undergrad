import os
from flask import Flask, render_template, request, send_from_directory, jsonify, abort
import file_processor


class SismicDataManager:
    def __init__(self, raw_dir="raw", processed_dir="eventos_procesados"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.cached_data = None
        self.precompute_data()

    def precompute_data(self):
        file_processor.copy_and_group_sac_files(
            self.raw_dir, self.processed_dir, filter_single_file_stations=False
        )

        # Get available sensors and components from processed files
        available_sensors, available_components = file_processor.get_available_filters(
            self.raw_dir
        )

        self.cached_data = {
            "sensors": available_sensors,
            "components": available_components,
        }

    def get_filtered_data(self, sensors=None, components=None, filter_stations=False):
        # List all files in the processed directory
        all_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".png")]

        filtered_data = {}
        for filename in all_files:
            try:
                # Split filename to extract details
                parts = filename.split(".")
                if len(parts) >= 4:
                    red, estacion, file_sensor, file_component = parts[:4]

                    # Apply filters
                    sensor_match = not sensors or file_sensor in sensors
                    component_match = not components or file_component in components

                    if sensor_match and component_match:
                        station_key = f"{red}.{estacion}"
                        if station_key not in filtered_data:
                            filtered_data[station_key] = []
                        filtered_data[station_key].append(filename)
            except Exception as e:
                print(f"Error processing filename {filename}: {e}")

        # Filter stations with multiple files if required
        if filter_stations:
            filtered_data = {k: v for k, v in filtered_data.items() if len(v) > 1}

        return filtered_data


app = Flask(__name__)
data_manager = SismicDataManager()


@app.route("/")
def index():
    selected_sensors = request.args.getlist("sensors")
    selected_components = request.args.getlist("components")
    filter_stations = (
        request.args.get("filter_stations", default="false").lower() == "true"
    )

    filtered_stations = data_manager.get_filtered_data(
        sensors=selected_sensors or None,
        components=selected_components or None,
        filter_stations=filter_stations,
    )

    return render_template(
        "index.html",
        estaciones=filtered_stations,
        available_sensors=data_manager.cached_data["sensors"],
        available_components=data_manager.cached_data["components"],
        selected_sensors=selected_sensors,
        selected_components=selected_components,
        filter_stations=filter_stations,
    )


@app.route("/images/<path:filename>")
def serve_image(filename):
    png_path = os.path.join("eventos_procesados", filename)
    if not os.path.exists(png_path):
        abort(404)
    return send_from_directory("eventos_procesados", filename)


@app.route("/apply_changes", methods=["POST"])
def apply_changes():
    data = request.get_json()
    for estacion, acciones in data.items():
        for archivo, accion in acciones.items():
            ruta_archivo = os.path.join("eventos_procesados", archivo)
            if accion == "delete" and os.path.exists(ruta_archivo):
                os.remove(ruta_archivo)

    return jsonify({"message": "Changes applied successfully"}), 200


if __name__ == "__main__":
    app.run(debug=True)
