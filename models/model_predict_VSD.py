import librosa.display
import matplotlib.pyplot as plt
import librosa
import wavelet_denoise
from scipy import signal
import numpy as np
import pandas as pd
import more_itertools as mit
import copy
import pickle
from keras.models import load_model
import sys

def range_size(x):
    return np.max(abs(x))-np.min(abs(x))
#中值濾波器
def band_pass_filter(original_signal, order, fc1, fc2, fs):
    """
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 濾波後的音頻數據
    """
    sos = signal.butter(N=order, Wn=[fc1, fc2], btype='band', fs=fs, output='sos')
    new_signal = signal.sosfilt(sos, original_signal)
    return new_signal
#切割音框
def feature_extraction_origin(files, sample_rate):
    '''
        files : 資料data
        sample_rate : 取樣率
        return : 分段的index值
    '''
    # turn the wave's data to array
    all_data = []
    CutTimeDef = 0.02
    framerate = sample_rate  # 收集頻率
    nframes = len(files)  # 資料長度
    CutFrameNum = framerate * CutTimeDef
    wave_data = np.array(files)
    # calculate the time bar
    time = np.arange(0, nframes) * (1.0 / framerate)
    # 要計算總共要製作幾個 slide windows，當無法完全整除時，將無條件捨去取正整數
    slide_sec = 0.01
    slide_windows_num = int(((len(time) - (CutFrameNum)) / (framerate * slide_sec)) + 1)
    # 創建一個數組來定義數據的分段，開始時間與結束時間
    start_time = np.arange(start=0, stop=(slide_windows_num - 1) * (framerate * slide_sec)+1, step=((framerate * slide_sec)))
    stop_time = np.arange(start=(CutFrameNum), stop=((slide_windows_num - 1) * (framerate * slide_sec)) + CutFrameNum + 1,
                          step=((framerate * slide_sec)))
    # 存取開始與結束
    all_array = np.array([start_time, stop_time])
    all_array = all_array.T
    # 分段存取
    for i in range(len(all_array[:, 0])):
        start_time1 = all_array[i, 0]
        stop_time1 = all_array[i, 1]
        all_data.append(wave_data[int(start_time1):int(stop_time1)])
    all_data1 = pd.DataFrame(np.array(all_data))
    return all_data1
#移除s1部分
def remove_S1(data, No_S1_point):
    '''
    data : 輸入預處理data
    No_S1_point : 不屬於s1的數值範圍 (array)
    return : 將屬於s1的部分歸0後的data
    '''
    original_data = copy.deepcopy(data)
    data[0: int(No_S1_point[0,0])] = 0
    for i in range(No_S1_point.shape[0] - 1):
        data[int(No_S1_point[i, 1]) : int(No_S1_point[i+1, 0])] = 0
    data[int(No_S1_point[-1, 1]) : 10000] = 0
    return data
def S1_segmentation(x, model):
    '''
        x : 檔案名稱
        model : 切割 s1 K-means 的 model
        return :
                S1_data : 屬於S1的data
                center_number_new : 不屬於s1的index
    '''
    # 讀取音頻
    audio_data, fs = librosa.load( x , sr=2000, duration=10)
    #wavelet denoise
    audio_data1 = wavelet_denoise.paper_wavelet(audio_data)
    # title_name = FileName
    # 數字濾波
    audio_data2 = band_pass_filter(original_signal=audio_data1, order=2, fc1=20, fc2=150, fs=fs)
    down_sample_rate = 1000
    # 降採樣
    down_sample_audio_data = librosa.resample(audio_data2.T, orig_sr=fs, target_sr=down_sample_rate).T
    # 標準化
    down_sample_audio_data1 = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
    # 擷取資料的統計量
    x_1 = down_sample_audio_data1
    data_test = feature_extraction_origin(down_sample_audio_data1, down_sample_rate)
    data_std1 = data_test.apply(np.std, axis=1).to_numpy().reshape(data_test.shape[0], 1)
    data_range1 = data_test.apply(range_size, axis=1).to_numpy().reshape(data_test.shape[0], 1)
    data_all_feature1 = np.concatenate((data_std1, data_range1), axis=1)
    # 進行預測
    labels1 = model.predict(data_all_feature1)
    detect_number = []
    for i in range(0, labels1.shape[0]):
        if (np.sum(labels1[i:(i + 3)]) >= 3):
            detect_number.append(i)
    detect_number = np.array(detect_number, dtype=float)  # 連續三個點要被偵測到
    each_group = [list(group) for group in mit.consecutive_groups(detect_number)]
    center_number = []
    for j in range(0, len(each_group)):
        each_group[j] = each_group[j] + [each_group[j][len(each_group[j]) - 1] + 1,
                                         each_group[j][len(each_group[j]) - 1] + 2]
        start_value = 20 + (each_group[j][0] - 1) * 10
        end_value = 20 + (each_group[j][int(len(each_group[j])) - 1] - 1) * 10
        int_max_value = np.argmax(abs(x_1[int(start_value):int(end_value)])) + int(start_value)
        center_number.append(int_max_value)  # 取中心

    center_number_new = np.array(center_number, dtype=float)
     # 400 ms 內的最大值
    max_value_x1 = []
    s1_s2_int = 50
    s1_s1_int = 400
    for i in range(0, len(center_number_new)):
        if center_number_new[i] < s1_s1_int:
            max_value_x1.append(
                np.argmax(abs(x_1[int(center_number_new[i]):int(center_number_new[i] + s1_s1_int)])) + int(
                    center_number_new[i]))
        else:
            max_value = np.argmax(
                abs(x_1[int(center_number_new[i] - s1_s1_int):int(center_number_new[i] + s1_s1_int)])) + int(
                center_number_new[i] - s1_s1_int)
            if i != 0 and abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) <= 250:
                max_value_renew = int((max_value_x1[(len(max_value_x1) - 1)] + max_value) / 2)
                max_value_x1[(len(max_value_x1) - 1)] = max_value_renew
            elif i != 0 and abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) >= 1400:
                max_value_renew = int(abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) / 2) + max_value_x1[
                    (len(max_value_x1) - 1)]
                max_value_renew = np.argmax(abs(x_1[int(max_value_renew - 100):int(max_value_renew + 100)]))
                all_max_value = [max_value_x1[(len(max_value_x1) - 1)]] + [max_value_renew] + [max_value]
                max_value_x1 = max_value_x1 + all_max_value
            else:
                max_value_x1.append(max_value)
    max_value_x1 = np.unique(max_value_x1)

    center_number = max_value_x1

    center_number_new_cut = []
    center_number_diff = np.diff(center_number)
    cut_prob = 0
    for i in range(0, len(center_number_diff)):
        if 250 < center_number_diff[i] < 1400:
            renew_center_number = [center_number[i], center_number[(i + 1)]]
            center_number_new_cut.append(renew_center_number)
        elif center_number_diff[i] >= 1400:
            print("s1 間隔過大")
            cut_prob += 1
        else:
            print("s1 間隔過小")
            cut_prob += 1

    # 計算心率
    between = []
    for i in range(0, (len(center_number_new_cut) - 1)):
        between.append((center_number_new_cut[i][1] - center_number_new_cut[i][0]) * 1 / 1000)
    heart_rate = 60 / np.mean(between)

    # 計算分割點
    center_number_new = np.array(center_number_new_cut, dtype=float)
    for j in range(0, center_number_new.shape[0]):
        if j == 0 and center_number_new[0][0] < 50:
            center_number_new[0][0] = np.argmax(
                abs(x_1[int(center_number_new[j][0]):int(center_number_new[j][0] + 100)])) + int(
                center_number_new[0][0])
            center_number_new[0][1] = np.argmax(
                abs(x_1[int(center_number_new[j][1] - 50):int(center_number_new[j][1] + 50)])) + int(
                center_number_new[j][1] - 50)
        else:
            center_number_new[j][0] = np.argmax(
                abs(x_1[int(center_number_new[j][0] - 50):int(center_number_new[j][0] + 50)])) + int(
                center_number_new[j][0] - 50) + 50
            center_number_new[j][1] = np.argmax(
                abs(x_1[int(center_number_new[j][1] - 50):int(center_number_new[j][1] + 50)])) + int(
                center_number_new[j][1] - 50) - 50
    S1_data = np.array([])
    for i in range(0, len(center_number_new) - 1):
       S1_data = np.append(S1_data, down_sample_audio_data1[ int(center_number_new[i][1]) : int(center_number_new[i+1][0])] )

    return S1_data, center_number_new
def S2_segmentation(x, No_S1_point, model):
    '''
        x : 檔案名稱
        No_s1_point : 不屬於S1的資料點位置
        model : S2 K-means model
        return : 移除s1及s2後的data
    '''
    # 讀取音頻
    audio_data, fs = librosa.load(x, sr=2000, duration=10)
    # wavelet denoise
    audio_data1 = wavelet_denoise.paper_wavelet(audio_data)
    # title_name = FileName
    # 數字濾波
    audio_data2 = band_pass_filter(original_signal=audio_data1, order=2, fc1=20, fc2=150, fs=fs)
    down_sample_rate = 1000
    # 降採樣
    down_sample_audio_data = librosa.resample(audio_data2.T, orig_sr=fs, target_sr=down_sample_rate).T
    # 標準化
    down_sample_audio_data1 = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
    # 擷取資料的統計量
    x_1 = down_sample_audio_data1
    down_sample_rate = 1000
    down_sample_audio_data1 = remove_S1(x_1, No_S1_point)
    data_test = feature_extraction_origin(down_sample_audio_data1, down_sample_rate)
    data_std1 = data_test.apply(np.std, axis=1).to_numpy().reshape(data_test.shape[0], 1)
    data_range1 = data_test.apply(range_size, axis=1).to_numpy().reshape(data_test.shape[0], 1)
    data_all_feature1 = np.concatenate((data_std1, data_range1), axis=1)
    # 進行預測
    labels1 = model.predict(data_all_feature1)
    detect_number = []
    for i in range(0, labels1.shape[0]):
        if (np.sum(labels1[i:(i + 3)]) >= 3):
            detect_number.append(i)
    detect_number = np.array(detect_number, dtype=float)  # 連續三個點要被偵測到
    each_group = [list(group) for group in mit.consecutive_groups(detect_number)]
    center_number = []
    for j in range(0, len(each_group)):
        each_group[j] = each_group[j] + [each_group[j][len(each_group[j]) - 1] + 1,
                                         each_group[j][len(each_group[j]) - 1] + 2]
        start_value = 20 + (each_group[j][0] - 1) * 10
        end_value = 20 + (each_group[j][int(len(each_group[j])) - 1] - 1) * 10
        int_max_value = np.argmax(abs(x_1[int(start_value):int(end_value)])) + int(start_value)
        center_number.append(int_max_value)  # 取中心

    center_number_new = np.array(center_number, dtype=float)
     # 400 ms 內的最大值
    max_value_x1 = []
    s1_s2_int = 50
    s1_s1_int = 400
    for i in range(0, len(center_number_new)):
        if center_number_new[i] < s1_s1_int:
            max_value_x1.append(
                np.argmax(abs(x_1[int(center_number_new[i]):int(center_number_new[i] + s1_s1_int)])) + int(
                    center_number_new[i]))
        else:
            max_value = np.argmax(
                abs(x_1[int(center_number_new[i] - s1_s1_int):int(center_number_new[i] + s1_s1_int)])) + int(
                center_number_new[i] - s1_s1_int)
            if i != 0 and abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) <= 250:
                max_value_renew = int((max_value_x1[(len(max_value_x1) - 1)] + max_value) / 2)
                max_value_x1[(len(max_value_x1) - 1)] = max_value_renew
            elif i != 0 and abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) >= 1400:
                max_value_renew = int(abs(max_value_x1[(len(max_value_x1) - 1)] - max_value) / 2) + max_value_x1[
                    (len(max_value_x1) - 1)]
                max_value_renew = np.argmax(abs(x_1[int(max_value_renew - 100):int(max_value_renew + 100)]))
                all_max_value = [max_value_x1[(len(max_value_x1) - 1)]] + [max_value_renew] + [max_value]
                max_value_x1 = max_value_x1 + all_max_value
            else:
                max_value_x1.append(max_value)
    max_value_x1 = np.unique(max_value_x1)

    center_number = max_value_x1

    center_number_new_cut = []
    center_number_diff = np.diff(center_number)
    cut_prob = 0
    for i in range(0, len(center_number_diff)):
        if 250 < center_number_diff[i] < 1400:
            renew_center_number = [center_number[i], center_number[(i + 1)]]
            center_number_new_cut.append(renew_center_number)
        elif center_number_diff[i] >= 1400:
            print("s1 間隔過大")
            cut_prob += 1
        else:
            print("s1 間隔過小")
            cut_prob += 1

    # 計算心率
    between = []
    for i in range(0, (len(center_number_new_cut) - 1)):
        between.append((center_number_new_cut[i][1] - center_number_new_cut[i][0]) * 1 / 1000)
    heart_rate = 60 / np.mean(between)

    # 計算分割點
    center_number_new = np.array(center_number_new_cut, dtype=float)
    for j in range(0, center_number_new.shape[0]):
        if j == 0 and center_number_new[0][0] < 50:
            center_number_new[0][0] = np.argmax(
                abs(x_1[int(center_number_new[j][0]):int(center_number_new[j][0] + 100)])) + int(
                center_number_new[0][0])
            center_number_new[0][1] = np.argmax(
                abs(x_1[int(center_number_new[j][1] - 50):int(center_number_new[j][1] + 50)])) + int(
                center_number_new[j][1] - 50)
        else:
            center_number_new[j][0] = np.argmax(
                abs(x_1[int(center_number_new[j][0] - 50):int(center_number_new[j][0] + 50)])) + int(
                center_number_new[j][0] - 50) + 50
            center_number_new[j][1] = np.argmax(
                abs(x_1[int(center_number_new[j][1] - 50):int(center_number_new[j][1] + 50)])) + int(
                center_number_new[j][1] - 50) - 50

    down_sample_audio_data1[0: int(center_number_new[0, 0])] = 0
    for i in range(center_number_new.shape[0] - 1):
        down_sample_audio_data1[int(center_number_new[i, 1]): int(center_number_new[i + 1, 0])] = 0
    down_sample_audio_data1[int(center_number_new[-1, 1]): 10000] = 0

    remove_0_data = np.array([])
    for i in range(0, down_sample_audio_data1.shape[0]):
        if(down_sample_audio_data1[i] != 0):
            remove_0_data= np.append(remove_0_data, down_sample_audio_data1[i])

    return remove_0_data
def plt_original_signel(audio_path, imgName):
    data, fs = librosa.load(audio_path, sr=2000)
    audio_data1 = wavelet_denoise.paper_wavelet(data)
    # title_name = FileName
    # 數字濾波
    audio_data2 = band_pass_filter(original_signal=audio_data1, order=2, fc1=20, fc2=150, fs=fs)
    down_sample_rate = 1000
    # 降採樣
    down_sample_audio_data = librosa.resample(audio_data2.T, orig_sr=fs, target_sr=down_sample_rate).T
    # 標準化
    down_sample_audio_data1 = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))

    plt.figure(figsize=(12, 6))
    plt.plot(down_sample_audio_data1)
    plt.title('Heart Sound Chart')
    plt.savefig("./images/HeartSound/" + imgName)
def extract_feature(file_name, model, model_S2):
    S1_data, no_s1_center = S1_segmentation(file_name, model)
    S2_data = S2_segmentation(file_name, no_s1_center, model_S2)
    S2_data = np.array(S2_data)
    resize_S2_data = np.resize(S2_data, (10000,))
    S2_mfccs = np.mean(librosa.feature.mfcc(y=resize_S2_data, sr=1000, n_mfcc=20).T, axis=0)
    # S2_mfccs = S2_mfccs / np.max(np.abs(S2_mfccs))
    S2_feature = np.array(S2_mfccs).reshape([-1, 1])
    return S2_feature
if __name__ == '__main__':
    audio_path = sys.argv[1]
    imgName = sys.argv[2].strip(".wav") + ".png"
    plt_original_signel(audio_path, imgName)
    model_file_name_S1 = './models/Kmeans_model_S1.h5'
    with open(model_file_name_S1, 'rb') as f:
        kmeans_model_S1 = pickle.load(f)
    # Extract_S1_file(kmeans_model_S1)
    model_file_name_S2 = './models/Kmeans_model_S2.h5'
    with open(model_file_name_S2, 'rb') as f:
        kmeans_model_S2 = pickle.load(f)
    mfccs = extract_feature(audio_path, kmeans_model_S1, kmeans_model_S2)
    model = load_model('./models/CNN_20_VSD_normal.hdf5')
    result = model.predict(mfccs.reshape([1,20,1]))
    max = np.argmax(result)
    sys.stdout.write(str(max))
