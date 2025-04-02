import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import time

inputFilePath = 'HSI/Moffett_Field_8.mat'
inputData = loadmat(inputFilePath)
inputImage = inputData['Xim'][:255, :255, :]

start_time = time.time()

BIT_DEPTH = 8
NUM_TONES = float(2**BIT_DEPTH - 1)
FLOATING_POINT_REPRESENTATION = 'float32'
IMAGE_INTEGER_REPRESENTATION = 'uint8'

originalImage = inputImage.astype(FLOATING_POINT_REPRESENTATION)
normalizedImage = originalImage / NUM_TONES

num_pixels, num_bands = normalizedImage.shape[0] * normalizedImage.shape[1], normalizedImage.shape[2]
normalizedImage = normalizedImage.reshape(num_pixels, num_bands)

pca = PCA(n_components=1)
mms = MinMaxScaler(feature_range=(0, NUM_TONES))

reducedImage = pca.fit_transform(normalizedImage)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

compressed_data = mms.fit_transform(reducedImage).astype(IMAGE_INTEGER_REPRESENTATION)
JPEG2000Encoding = cv2.imencode(".jpg", compressed_data.astype(IMAGE_INTEGER_REPRESENTATION))[1]

JPEG2000Decoding = mms.inverse_transform(cv2.imdecode(JPEG2000Encoding, cv2.IMREAD_UNCHANGED).astype(FLOATING_POINT_REPRESENTATION))
reconstructedImage = pca.inverse_transform(JPEG2000Decoding) * NUM_TONES

reconstructedImage = reconstructedImage.reshape(normalizedImage.shape)

restoredImage = reconstructedImage.reshape(originalImage.shape)

data_to_save = {'pca_jpeg2k': restoredImage}
savemat('REC/pca+jpeg2000(MF).mat', data_to_save)

end_time = time.time()


def sam(x, y):
    num = np.sum(np.multiply(x, y), 2)
    den = np.sqrt(np.multiply(np.sum(x**2, 2), np.sum(y**2, 2)))
    sam = np.sum(np.degrees(np.arccos(num / den))) / (x.shape[0]*x.shape[1])
    return sam

def psnr(x,y):
    bands = x.shape[2]
    x = np.reshape(x, [-1, bands])
    y = np.reshape(y, [-1, bands])
    msr = np.mean((x-y)**2, 0)
    maxval = np.max(y, 0)**2
    return np.mean(10*np.log10(maxval/msr))

def rmse(x, y):
    aux = np.sum(np.sum((x-y)**2, 0),0) / (x.shape[0]*x.shape[1])
    rmse_per_band = np.sqrt(aux)
    rmse_total = np.sqrt(np.sum(aux) / x.shape[2])
    return rmse_total

# 计算指标
total_time = end_time - start_time
sam_value = sam(restoredImage, originalImage)
rmse_value = rmse(restoredImage, originalImage)
psnr_value = psnr(restoredImage, originalImage)


print("SAM:", sam_value)
print("RMSE:", rmse_value)
print("PSNR:", psnr_value)
print(f"Total execution time: {total_time} seconds")

