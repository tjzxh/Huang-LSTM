from tensorflow import keras
from keras.layers import Dense, LSTM, ReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn import preprocessing

# Sort Reconstructed NGSIM data
origin_data = np.loadtxt('final_DATA.txt')
ind = np.lexsort((origin_data[:, 1], origin_data[:, 0]))
origin_data = origin_data[ind, :]
data_size = origin_data.shape[0]
veh_id, frame_id, speed, gap, speed_diff = origin_data[:, 0], origin_data[:, 1], origin_data[:, 4], origin_data[:,
                                                                                                    10], origin_data[:,
                                                                                                         11]
# Standardization
vs_array = np.hstack((speed.reshape(data_size, 1), gap.reshape(data_size, 1)))
vs_scaler = preprocessing.StandardScaler().fit(vs_array)
vd_scaler = preprocessing.StandardScaler().fit(speed_diff.reshape(data_size, 1))
print("vs_scaler mean:", vs_scaler.mean_)
print("vs_scaler std:", vs_scaler.scale_)
vs_scaled = vs_scaler.transform(vs_array)
vd_scaled = vd_scaler.transform(speed_diff.reshape(data_size, 1))
all_veh_id = list(set(veh_id.tolist()))
all_veh_id.sort()
end_veh_id = all_veh_id[650] if len(all_veh_id) > 650 else all_veh_id[-1]
all_input, all_output = [], []
print("Experiment starts")
for i in range(int(origin_data.shape[0]) - 50):
    if origin_data[i, 0] <= end_veh_id and origin_data[i, 0] == origin_data[i + 49, 0] and origin_data[i + 50, 1] - \
            origin_data[i, 1] == 50:
        all_input.append(np.hstack((vs_scaled[i:i + 50, :], vd_scaled[i:i + 50, :])))
        all_output.append(vs_scaled[i + 50, :])
print("Data processing over")
print("Data num is:", max(np.where(veh_id == end_veh_id)[0]))
all_input, all_output = np.array(all_input), np.array(all_output)

# Train Huang-LSTM for CF
neuron, time_step = 32, 50
model = keras.Sequential()
model.add(LSTM(neuron, return_sequences=True, input_shape=(time_step, 3), activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=True, activation='tanh'))
model.add(LSTM(neuron, return_sequences=False, activation='tanh'))
model.add(Dense(neuron))
model.add(ReLU(negative_slope=0.9))
model.add(Dense(2))
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9)
model.compile(optimizer=sgd, loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(all_input, all_output, batch_size=100, epochs=20, callbacks=[early_stopping], verbose=1,
          validation_split=0.3)

# Save the trained model
model_name = "Huang_LSTM"
model.save(model_name)
