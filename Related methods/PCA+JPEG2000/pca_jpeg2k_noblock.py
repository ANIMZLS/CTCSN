from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import time

inputFilePath = 'HSI/HyspIRI_2_77.mat'
inputData = loadmat(inputFilePath)
inputImage = inputData['Xim']

start_time = time.time()

BIT_DEPTH = 8
NUM_TONES = float(2**BIT_DEPTH - 1)
FLOATING_POINT_REPRESENTATION = 'float32'
IMAGE_INTEGER_REPRESENTATION = 'float32'

originalImage = inputImage.astype(FLOATING_POINT_REPRESENTATION)
normalizedImage = originalImage / NUM_TONES

# 将三维数组转换为二维数组
num_pixels, num_bands = normalizedImage.shape[0] * normalizedImage.shape[1], normalizedImage.shape[2]
normalizedImage = normalizedImage.reshape(num_pixels, num_bands)

# PCA和MinMaxScaler初始化
pca = PCA(n_components=2)
mms = MinMaxScaler(feature_range=(0, NUM_TONES))

reducedImage = pca.fit_transform(normalizedImage)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# COMPRESS
compressed_data = mms.fit_transform(reducedImage).astype(IMAGE_INTEGER_REPRESENTATION)
compressed_image = Image.fromarray(compressed_data.astype('uint8'))

# DECOMPRESS
decompressed_data = mms.inverse_transform(np.array(compressed_image)).astype(FLOATING_POINT_REPRESENTATION)
reconstructedImage = pca.inverse_transform(decompressed_data) * NUM_TONES

# 将结果数组还原回原始形状
reconstructedImage = reconstructedImage.reshape(normalizedImage.shape)

# 将二维数组还原为三维数组
restoredImage = reconstructedImage.reshape(originalImage.shape)

# 假设 `reconstructedImage` 是解压缩后的数据
data_to_save = {'pca_jpeg2k': restoredImage}
savemat('REC/pca+jpeg2000(U).mat', data_to_save)

end_time = time.time()
# 计算指标
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

