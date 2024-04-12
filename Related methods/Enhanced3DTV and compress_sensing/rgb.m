addpath(genpath('测试图片/pca_jpeg2k'));
addpath(genpath('测试图片/CTCSN'));
addpath(genpath('测试图片/BTCNet'));
addpath(genpath('测试图片/DCSN'));
addpath(genpath('测试图片/mat'));
load('HyspIRI_2_77_BTC.mat');  % 替换为你的高光谱图像文件名

band_set=[25 15 6];
temp_show=BTC(:,:,band_set);
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=normColor(temp_show);
figure;
imshow(temp_show);
%imwrite(temp_show, "C:/Users/Z.LS/Desktop/数据对比/定性分析/CTCSN(U).png", 'png');
