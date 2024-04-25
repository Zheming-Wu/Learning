# Author: Wu Zheming


import numpy as np
import paho.mqtt.client as mqtt
import time

broker = "broker.emqx.io"
port = 1883
topic = "/data/movement_simulation"

# Initialize variables
t_data = []
x_data = []
y_data = []
vx_data = []
vy_data = []

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic, qos=2)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        print("Receiving data")
        data = msg.payload.decode('utf-8').split('#')
        t_data.append(float(data[0]))
        x_data.append(float(data[1]))
        y_data.append(float(data[2]))
        vx_data.append(float(data[3]))
        vy_data.append(float(data[4]))
    except Exception as e:
        print(f"Error parsing MQTT message: {e}")

def save_to_numpy():
    global t_data, x_data, y_data, vx_data, vy_data
    t_array = np.array(t_data)
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    vx_array = np.array(vx_data)
    vy_array = np.array(vy_data)
    return t_array, x_array, y_array, vx_array, vy_array

def mqtt_sub(client):
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, 10)
    client.loop_start() 

def main():
    client = mqtt.Client()
    mqtt_sub(client)
    
    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        t_array, x_array, y_array, vx_array, vy_array = save_to_numpy()
        print("Data saved to numpy arrays")

if __name__ == '__main__':
    main()
