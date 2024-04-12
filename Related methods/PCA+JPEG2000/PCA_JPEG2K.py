import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import time

inputFilePath = 'HSI/California_41.mat'
inputData = loadmat(inputFilePath)
inputImage = inputData['Xim']

start_time = time.time()

BIT_DEPTH = 8
NUM_TONES = float(2**BIT_DEPTH - 1)
FLOATING_POINT_REPRESENTATION = 'float32'
IMAGE_INTEGER_REPRESENTATION = 'uint8'

originalImage = inputImage.astype(FLOATING_POINT_REPRESENTATION)
normalizedImage = originalImage / NUM_TONES

# 将三维数组转换为二维数组
num_pixels, num_bands = normalizedImage.shape[0] * normalizedImage.shape[1], normalizedImage.shape[2]
normalizedImage = normalizedImage.reshape(num_pixels, num_bands)

# 设置块的大小
block_size = 256

# 获取块的数量
num_blocks = num_pixels // block_size

# 初始化结果数组
reconstructedImage_blocks = np.zeros_like(normalizedImage)

# PCA和MinMaxScaler初始化
pca = PCA(n_components=1)
mms = MinMaxScaler(feature_range=(0, NUM_TONES))

reducedImage = pca.fit_transform(normalizedImage)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# 分块处理
for i in range(num_blocks):
    start_idx = i * block_size
    end_idx = (i + 1) * block_size
    block_data = normalizedImage[start_idx:end_idx, :]

    # COMPRESS
    reducedImage_block = pca.fit_transform(block_data)
    JPEG2000Encoding_block = cv2.imencode(".jpg", mms.fit_transform(reducedImage_block).astype(IMAGE_INTEGER_REPRESENTATION))[1]

    # DECOMPRESS
    JPEG2000Decoding_block = mms.inverse_transform(cv2.imdecode(JPEG2000Encoding_block, cv2.IMREAD_UNCHANGED).astype(FLOATING_POINT_REPRESENTATION))
    reconstructedImage_block = pca.inverse_transform(JPEG2000Decoding_block) * NUM_TONES

    # 将块的结果存储到结果数组中
    reconstructedImage_blocks[start_idx:end_idx, :] = reconstructedImage_block

# 将结果数组还原回原始形状
reconstructedImage = reconstructedImage_blocks.reshape(normalizedImage.shape)

# 将二维数组还原为三维数组
restoredImage = reconstructedImage.reshape(originalImage.shape)
end_time = time.time()

# 假设 `reconstructedImage` 是解压缩后的数据
data_to_save = {'pca_jpeg2k': restoredImage}
savemat('REC/pca+jpeg2000(U).mat', data_to_save)

def sam(x, y):
    '''
    num = sum(x .* y, 3);
    den = sqrt(sum(x.^2, 3) .* sum(y.^2, 3));
    sam = sum(sum(acosd(num ./ den)))/(n_samples);
    '''
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
sam_value = sam(restoredImage, originalImage)
rmse_value = rmse(restoredImage, originalImage)
psnr_value = psnr(restoredImage, originalImage)
total_time = end_time - start_time

print("SAM:", sam_value)
print("RMSE:", rmse_value)
print("PSNR:", psnr_value)
print(f"Total execution time: {total_time} seconds")

# 计算压缩比
original_size = inputImage.nbytes
compressed_size = sum(len(JPEG2000Encoding_block) for i in range(num_blocks))
compression_ratio = original_size / compressed_size
print("Original Size (bytes):", original_size)
print("Compressed Size (bytes):", compressed_size)
print("Compression Ratio:", compression_ratio)
