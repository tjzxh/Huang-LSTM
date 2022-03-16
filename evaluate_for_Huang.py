import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import numpy as np
from random import sample
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
# Sort Reconstructed NGSIM data
origin_data = np.loadtxt('final_DATA.txt')
ind = np.lexsort((origin_data[:, 1], origin_data[:, 0]))
origin_data = origin_data[ind, :]
data_size = origin_data.shape[0]
veh_id, frame_id, speed, gap, speed_diff, acce = origin_data[:, 0], origin_data[:, 1], origin_data[:, 4], origin_data[:,
                                                                                                          10], origin_data[
                                                                                                               :,
                                                                                                               11], origin_data[
                                                                                                                    :,
                                                                                                                    5]
# Standardization
# Load Scaler
input_scaler = joblib.load("input_scaler.save")
output_scaler = joblib.load("output_scaler.save")
all_veh_id = list(set(veh_id.tolist()))
all_veh_id.sort()
test_veh_id = sample(all_veh_id, 10)
print("Experiment starts")
# Load the trained model
model_name = "Huang_LSTM_0312"
model = load_model(model_name)

for ind in test_veh_id:
    test_input, test_output = [], []
    subject_ind = np.where(origin_data[:, 0] == ind)
    subject_data = origin_data[subject_ind]
    for i in range(int(subject_data.shape[0]) - 50):
        if subject_data[i, 0] == subject_data[i + 49, 0] and subject_data[i + 50, 1] - subject_data[i, 1] == 50:
            test_input.append(subject_data[i:i + 50, [4, 10, 11]])
            test_output.append(subject_data[i + 50, 4])
    # Test the Model
    test_input, test_output = np.array(test_input), np.array(test_output)
    flat_input = test_input.flatten()
    input_scaled = input_scaler.transform(flat_input.reshape((test_output.shape[0] * 50, 3)))
    output_scaled = output_scaler.transform(test_output.reshape(test_output.shape[0], 1))
    flat_input = input_scaled.flatten()
    final_input = flat_input.reshape((test_output.shape[0], 50, 3))
    prediction = model.predict(final_input)
    real_pred = output_scaler.inverse_transform(prediction)
    plt.figure()
    plt.plot(range(test_output.shape[0]), test_output, label='real data', c='r')
    plt.plot(range(test_output.shape[0]), real_pred, label='prediction', c='b')
    plt.title('veh ' + str(int(ind)))
    plt.xlabel('time step (0.1s)')
    plt.ylabel('velocity (m/s)')
    plt.legend()
    plt.show()
