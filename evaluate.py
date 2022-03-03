from tensorflow import keras
from keras.models import load_model
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

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
start_veh_id = all_veh_id[900] if len(all_veh_id) > 900 else all_veh_id[-1]
end_veh_id = all_veh_id[1200] if len(all_veh_id) > 1200 else all_veh_id[-1]
test_input, test_output = [], []
print("Experiment starts")
for i in range(int(origin_data.shape[0]) - 50):
    if start_veh_id <= origin_data[i, 0] <= end_veh_id and origin_data[i, 0] == origin_data[i + 49, 0] and origin_data[
        i + 50, 1] - origin_data[i, 1] == 50:
        test_input.append(np.hstack((vs_scaled[i:i + 50, :], vd_scaled[i:i + 50, :])))
        test_output.append(vs_scaled[i + 50, :])
print("Data processing over")
test_input, test_output = np.array(test_input), np.array(test_output)

# Load the trained model
model_name = "Huang_LSTM"
model = load_model(model_name)
scaled_prediction = model.predict(test_input)
prediction = vs_scaler.inverse_transform(scaled_prediction)
real_output = vs_scaler.inverse_transform(test_output)
# Calculation of Mean Squared Error (MSE)
mse_v = mean_squared_error(real_output[:, 0], prediction[:, 0])
mse_s = mean_squared_error(real_output[:, 1], prediction[:, 1])
print("mse_v = ", mse_v)
print("mse_s = ", mse_s)
# mse_v =  0.20071491697951135
# mse_s =  22.048061719914394
