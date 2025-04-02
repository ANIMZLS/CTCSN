load California_41;
inputImage = Xim;
start_time = tic;

BIT_DEPTH = 16;
NUM_TONES = 2^BIT_DEPTH - 1;
FLOATING_POINT_REPRESENTATION = 'single';
IMAGE_INTEGER_REPRESENTATION = 'uint16';

originalImage = single(inputImage);
normalizedImage = originalImage / NUM_TONES;

% 将三维数组转换为二维数组
[num_pixels, num_bands] = size(normalizedImage);
normalizedImage = reshape(normalizedImage, num_pixels, num_bands);

% 设置块的大小
block_size = 256;

% 获取块的数量
num_blocks = num_pixels / block_size;

% 初始化结果数组
reconstructedImage_blocks = zeros(size(normalizedImage), 'single');

% PCA和MinMaxScaler初始化
pca = pcares(normalizedImage, 1);
mms = MinMaxScaler('feature_range', [0, NUM_TONES]);

reducedImage = pca.transform(normalizedImage);
explained_variance_ratio = pca.MS.explained_variance_ratio;
disp(['解释方差比例: ', num2str(explained_variance_ratio)])

% 分块处理
for i = 1:num_blocks
    start_idx = (i - 1) * block_size + 1;
    end_idx = i * block_size;
    block_data = normalizedImage(start_idx:end_idx, :);

    % 压缩
    reducedImage_block = pca.transform(block_data);
    JPEG2000Encoding_block = imencode('.jpg', mms.fit_transform(reducedImage_block), 'jpg');

    % 解压缩
    JPEG2000Decoding_block = mms.inverse_transform(imdecode(JPEG2000Encoding_block, 'jpg'));
    reconstructedImage_block = pca.inverse_transform(JPEG2000Decoding_block) * NUM_TONES;

    % 将块的结果存储到结果数组中
    reconstructedImage_blocks(start_idx:end_idx, :) = reconstructedImage_block;
end

% 将结果数组还原回原始形状
reconstructedImage = reshape(reconstructedImage_blocks, size(normalizedImage));

% 将二维数组还原为三维数组
restoredImage = reshape(reconstructedImage, size(originalImage));
end_time = toc(start_time);

% 假设 `reconstructedImage` 是解压缩后的数据
data_to_save = struct('pca_jpeg2k', restoredImage);
save('REC/pca+jpeg2000(U).mat', '-struct', 'data_to_save');

% 计算指标
sam_value = sam(restoredImage, originalImage);
rmse_value = rmse(restoredImage, originalImage);
psnr_value = psnr(restoredImage, originalImage);
total_time = end_time;

disp(['SAM: ', num2str(sam_value)]);
disp(['RMSE: ', num2str(rmse_value)]);
disp(['PSNR: ', num2str(psnr_value)]);
disp(['总执行时间: ', num2str(total_time), ' 秒']);

% 计算压缩比
original_size = numel(inputImage) * 4; % 假设使用单精度浮点数（4字节每个元素）
compressed_size = sum(cellfun('length', JPEG2000Encoding_block));
compression_ratio = original_size / compressed_size;

disp(['原始大小（字节）: ', num2str(original_size)]);
disp(['压缩后大小（字节）: ', num2str(compressed_size)]);
disp(['压缩比: ', num2str(compression_ratio)]);

function value = sam(x, y)
    num = sum(x .* y, 3);
    den = sqrt(sum(x.^2, 3) .* sum(y.^2, 3));
    value = sum(sum(acosd(num ./ den))) / (size(x, 1) * size(x, 2));
end

function value = psnr(x, y)
    bands = size(x, 3);
    x = reshape(x, [], bands);
    y = reshape(y, [], bands);
    msr = mean((x - y).^2, 1);
    maxval = max(y, [], 1).^2;
    value = mean(10 * log10(maxval ./ msr));
end

function value = rmse(x, y)
    aux = sum(sum((x - y).^2, 1), 2) / (size(x, 1) * size(x, 2));
    rmse_per_band = sqrt(aux);
    value = sqrt(sum(aux) / size(x, 3));
end
