# Step 1: Install dependencies (run once in Colab)
!pip install azure-storage-blob
!pip install findspark pyspark paho-mqtt pandas joblib faker gradio
!apt-get update
!apt-get install openjdk-17-jdk -y
!wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
!tar xf /content/spark-3.5.1-bin-hadoop3.tgz

# -------------------------------------------------
# Step 2: Complete Streaming + Gradio Integration
import os
import findspark
findspark.init()

import json
import random
import threading
import time
from queue import Queue, Empty
from datetime import datetime, timedelta
import pandas as pd
import paho.mqtt.client as mqtt
from faker import Faker
import logging
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from azure.storage.blob import BlobServiceClient
import joblib
import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTT_Spark_Streaming")

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ['SPARK_HOME'] = "/content/spark-3.5.1-bin-hadoop3"
spark = SparkSession.builder.appName("HiveMQ_IoT_ML_Streaming").getOrCreate()

BROKER = "01b5269178e3464583fc069b9682daf6.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "Bharathnath"
PASSWORD = "#Bharath123"
TOPIC = "iot/failure"
fake = Faker()

iot_queue = Queue()
latest_predictions_queue = Queue()

device_mapping = {
    "Alaris GH": "Infusion Pump",
    "Baxter AK 96": "Dialysis Machine",
    "Baxter Flo-Gard": "Infusion Pump",
    "Datex Ohmeda S5": "Anesthesia Machine",
    "Drager Fabius Trio": "Anesthesia Machine",
    "Drager V500": "Patient Ventilator",
    "Fresenius 4008": "Dialysis Machine",
    "GE Aisys": "Anesthesia Machine",
    "GE Logiq E9": "Ultrasound Machine",
    "GE MAC 2000": "ECG Monitor",
    "GE Revolution": "CT Scanner",
    "Hamilton G5": "Patient Ventilator",
    "HeartStart FRx": "Defibrillator",
    "Lifepak 20": "Defibrillator",
    "NxStage System One": "Dialysis Machine",
    "Philips EPIQ": "Ultrasound Machine",
    "Philips HeartStrart": "Defibrillator",
    "Philips Ingenuity": "CT Scanner",
    "Phillips PageWriter": "ECG Monitor",
    "Puritan Bennett 980": "Patient Ventilator",
    "Siemens Acuson": "Ultrasound Machine",
    "Siemens S2000": "Ultrasound Machine",
    "Smiths Medfusion": "Infusion Pump",
    "Zoll R Series": "Defibrillator"
}

ClimateControl_list = ["Yes", "No"]
Location_list = [
    "Hospital A - Central Region","Hospital A - East Region","Hospital A - North Region","Hospital A - South Region","Hospital A - West Region",
    "Hospital B - Central Region","Hospital B - East Region","Hospital B - North Region","Hospital B - South Region","Hospital B - West Region",
    "Hospital C - Central Region","Hospital C - East Region","Hospital C - North Region","Hospital C - South Region","Hospital C - West Region",
    "Hospital D - Central Region","Hospital D - East Region","Hospital D - North Region","Hospital D - South Region","Hospital D - West Region",
    "Hospital E - Central Region","Hospital E - East Region","Hospital E - North Region","Hospital E - South Region","Hospital E - West Region",
    "Hospital F - Central Region","Hospital F - East Region","Hospital F - North Region","Hospital F - South Region","Hospital F - West Region",
    "Hospital G - Central Region","Hospital G - East Region","Hospital G - North Region","Hospital G - South Region","Hospital G - West Region",
    "Hospital H - Central Region","Hospital H - East Region","Hospital H - North Region","Hospital H - South Region","Hospital H - West Region"
]

AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=eliteforcework7884022717;AccountKey=2zBaEM4umwo3KW61B+N/45q+oH1oZwP5p0eMCt2BxJPLHv0XBmercIWtczi2f1ps/d5QnyTMA1IP+ASt2Rnocg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "azureml"
LOCAL_CSV_FILENAME = "predictions_output.csv"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(CONTAINER_NAME)

MODEL_PATH = "/content/xgboost_pipeline.pkl"
ml_pipeline = joblib.load(MODEL_PATH)
logger.info(f"Loaded ML pipeline from {MODEL_PATH}")

schema = StructType([
    StructField("DeviceType", StringType(), True),
    StructField("DeviceName", StringType(), True),
    StructField("RuntimeHours", DoubleType(), True),
    StructField("TemperatureC", DoubleType(), True),
    StructField("PressureKPa", DoubleType(), True),
    StructField("VibrationMM_S", DoubleType(), True),
    StructField("CurrentDrawA", DoubleType(), True),
    StructField("SignalNoiseLevel", DoubleType(), True),
    StructField("ClimateControl", StringType(), True),
    StructField("HumidityPercent", DoubleType(), True),
    StructField("Location", StringType(), True),
    StructField("OperationalCycles", IntegerType(), True),
    StructField("UserInteractionsPerDay", DoubleType(), True),
    StructField("LastServiceDate", StringType(), True),
    StructField("ApproxDeviceAgeYears", DoubleType(), True),
    StructField("NumRepairs", IntegerType(), True),
    StructField("ErrorLogsCount", IntegerType(), True)
])

def predict_udf(*cols):
    try:
        features = pd.DataFrame([cols], columns=[f.name for f in schema])
        prediction = ml_pipeline.predict(features)[0]
        return str(prediction)
    except Exception as e:
        return "Error"
predict = udf(predict_udf, StringType())

def random_service_date():
    days_ago = random.randint(0, 730)  # within last 2 years
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%d-%m-%Y")

def generate_record(device_name, device_type):
    return {
        "DeviceType": device_type,
        "DeviceName": device_name,
        "RuntimeHours": round(random.uniform(102.32, 9999.85), 2),
        "TemperatureC": round(random.uniform(16.07, 40), 2),
        "PressureKPa": round(random.uniform(90, 120), 2),
        "VibrationMM_S": round(random.uniform(0, 1), 3),
        "CurrentDrawA": round(random.uniform(0.1, 1.5), 3),
        "SignalNoiseLevel": round(random.uniform(0, 5), 2),
        "ClimateControl": random.choice(ClimateControl_list),
        "HumidityPercent": round(random.uniform(20, 70), 2),
        "Location": random.choice(Location_list),
        "OperationalCycles": random.randint(5, 11887),
        "UserInteractionsPerDay": round(random.uniform(0, 26.4), 2),
        "LastServiceDate": random_service_date(),
        "ApproxDeviceAgeYears": round(random.uniform(0.1, 35.89), 2),
        "NumRepairs": random.randint(0, 19),
        "ErrorLogsCount": random.randint(0, 22)
    }

def publish_simulated_continuous():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.connect(BROKER, PORT)
    client.loop_start()
    try:
        device_names = list(device_mapping.keys())
        while True:
            for device_name in device_names:
                device_type = device_mapping[device_name]
                record = generate_record(device_name, device_type)
                json_payload = json.dumps(record)
                client.publish(TOPIC, json_payload)
                print(f"📤 Published to {TOPIC}: {json_payload}")
                time.sleep(1)
    except Exception as e:
        logger.error("Publisher stopped: %s", str(e))
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Publishing stopped.")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    iot_queue.put(data)

def mqtt_subscribe():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_message = on_message
    client.connect(BROKER, PORT)
    client.subscribe(TOPIC)
    client.loop_forever()

def save_and_upload(records, filename=LOCAL_CSV_FILENAME):
    import csv
    if not records:
        return
    file_exists = os.path.exists(filename)
    keys = records[0].keys()
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)
    blob_name = f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "rb") as data:
        blob_container_client.upload_blob(blob_name, data, overwrite=True)
    logger.info(f"Uploaded file '{blob_name}' to Azure Blob Storage.")

batch_size = 100
batch_records = []

def streaming_processing_loop():
    global batch_records
    while True:
        if not iot_queue.empty():
            data = iot_queue.get()
            row = Row(**data)
            df = spark.createDataFrame([row], schema)
            df = df.withColumn("PredictedFailureRisk", predict(*df.columns))
            results = df.collect()
            for r in results:
                rec = r.asDict()
                print(rec)
                batch_records.append(rec)
            if len(batch_records) >= batch_size:
                save_and_upload(batch_records)
                latest_predictions_queue.put(batch_records.copy())
                batch_records.clear()

def get_latest_predictions():
    try:
        latest = latest_predictions_queue.get_nowait()
        total_records = len(latest)
        # Count by actual '0', '1', '2' (as strings)
        low_count = sum(1 for r in latest if str(r.get("PredictedFailureRisk")) == "0")
        medium_count = sum(1 for r in latest if str(r.get("PredictedFailureRisk")) == "1")
        high_count = sum(1 for r in latest if str(r.get("PredictedFailureRisk")) == "2")
        header = (f"Total Records: {total_records}\n"
                  f"Low Risk: {low_count}\n"
                  f"Medium Risk: {medium_count}\n"
                  f"High Risk: {high_count}\n\n"
                  "Latest Predictions (most recent batch):\n")
        formatted = "\n\n".join([str(rec) for rec in latest])
        return header + formatted
    except Empty:
        return "No new predictions yet."

iface = gr.Interface(
    fn=get_latest_predictions,
    inputs=[],
    outputs="textbox",
    live=False,
    title="Live Medical Equipment Failure Predictions",
    description="Click 'Run' to fetch the latest batch of streaming predictions."
)

threading.Thread(target=publish_simulated_continuous, daemon=True).start()
threading.Thread(target=mqtt_subscribe, daemon=True).start()
threading.Thread(target=streaming_processing_loop, daemon=True).start()
iface.launch()