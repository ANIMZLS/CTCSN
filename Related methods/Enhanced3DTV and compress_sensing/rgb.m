addpath(genpath('����ͼƬ/pca_jpeg2k'));
addpath(genpath('����ͼƬ/CTCSN'));
addpath(genpath('����ͼƬ/BTCNet'));
addpath(genpath('����ͼƬ/DCSN'));
addpath(genpath('����ͼƬ/mat'));
load('HyspIRI_2_77_BTC.mat');  % �滻Ϊ��ĸ߹���ͼ���ļ���

band_set=[25 15 6];
temp_show=BTC(:,:,band_set);
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
temp_show=normColor(temp_show);
figure;
imshow(temp_show);
%imwrite(temp_show, "C:/Users/Z.LS/Desktop/���ݶԱ�/���Է���/CTCSN(U).png", 'png');
