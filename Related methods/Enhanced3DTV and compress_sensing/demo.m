clc,clear; close all
clear all;clc;
addpath(genpath('compared method'));
addpath(genpath('quality assess'));
addpath(genpath('Enhanced3DTV'));
addpath(genpath('测试图片'));
seed = 2015; 
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

%% read data
load FtMyers_63
%x_dc = Xim(1:10,1:10,1:1);
x=Xim(:,:,:);
[w,h,s] = size(x);
%% Generate measurements
ratio  = 0.10; 
N      = h*w;
% A      = PermuteWHT2(N,s,ratio);
A      = PermuteWHT_partitioned(N,s,ratio);
AT     = A';
% A      = PermuteWHT2(N,s,ratio);
b      = A*x(:);

%% NCS
fprintf('=========== SLTNCS ============\n');
clear opts
opts.mu    = 2^12;
opts.beta  = 2^7; %2^5
opts.tol   = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = false;
t1 = cputime;
[U, out] = TVAL3(A,b,N,s,opts);
t1 = cputime - t1;
xrec_ncs = reshape(U,[w,h,s]);
i=1;[mpsnr(i),ssim(i),er(i),sam(i),rmse(i)] = msqia2(xrec_ncs,x);
% 保存为 .mat 文件
%save('SLTNCS_reconstructed_image.mat', 'xrec_ncs');


% % lrtdtv
fprintf('=========== JtenRe3DTV ============\n');
clear opts;
opts.maxIter = 100;
opts.tol     = 1e-9;
opts.trX     = x;
opts.gauss_frag = 1;
rk           = [ceil(h*0.6),ceil(w*0.6),6];
lam=0.001;
t2 = cputime;
x_rec_re= LrApprReTV(A,b,size(x),rk,lam,opts);
t2 = cputime - t2;
xrec_rettv=reshape(x_rec_re,size(x)); 
i=2;[mpsnr(i),ssim(i),er(i),sam(i),rmse(i)] = msqia2(xrec_rettv,opts.trX);
%fprintf('mpsnr = %f, ssim = %f, ergas =%f, sam =%f, rmse =%f.\n',mpsnr,ssim,ergas,sam,rmse);
% 保存为 .mat 文件
%save('jr3dtv_reconstructed_image.mat', 'xrec_rettv');
%band_set=[25 15 6];
%temp_show=xrec_rettv(:,:,band_set);
%normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
%temp_show=normColor(temp_show);
%figure;
%imshow(temp_show);
%imwrite(temp_show, "C:/Users/Z.LS/Desktop/数据对比/定性分析/JR3DTV(C).png", 'png');


gcs
fprintf('=========== Enhanced3DTV ============\n');
clear opts;
opts.maxIter = 100;
opts.tol     = 1e-6;
opts.trX     = x;
opts.gauss_frag = 1;
Rk           = 7;
weight       = [0.015,0.015,0.015];
t3 = cputime;
[x_gcs,x1,e,psnrpath] = EnhancedTV_CS(A, b, size(x), Rk, weight, opts);
t3 = cputime - t3;
xrec_ttv=reshape(x_gcs,[w,h,s]);
i=3;[mpsnr(i),ssim(i),er(i),sam(i),rmse(i)] = msqia2(xrec_ttv,opts.trX);
% 保存为 .mat 文件
%save('e3dtv_reconstructed_image.mat', 'xrec_ttv');
%band_set=[25 15 6];
%temp_show=xrec_ttv(:,:,band_set);
%normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
%temp_show=normColor(temp_show);
%figure;
%imshow(temp_show);
%imwrite(temp_show, "C:/Users/Z.LS/Desktop/数据对比/定性分析/E3DTV(MF).png", 'png');


% % lrtv
% fprintf('=========== lrtv ============\n');
% lambda=0.1; mu=[0.01 150 0.2]; iter=100;                                    
% xrec=funHSI(A,b,[w,h,s],lambda,mu,iter);
% xrec_lrtv=reshape(xrec,w,h,s);
% i=3;[mp(i),sm(i),er(i)] = msqia(xrec_lrtv/255,x/255);


%%sparcs
%fprintf('=========== sparcs ============\n');
%r=15;%15
%K = ceil(0.05*N*s); %0.02 
%[L,S,err]=sparcs(b, r, K, A, [w*h,s],'svds', 5e-4, 100, 1);
%xrec_sparcs=reshape((L+S),w,h,s);
%i=4;[mpsnr(i),ssim(i),er(i),sam(i),rmse(i)] = msqia2(xrec_sparcs,x);
% 保存为 .mat 文件
%save('sparcs_reconstructed_image.mat', 'xrec_sparcs');
%% KCS
%regularization parameter
%tau = ratio*max(abs(A'*b));%0.05
%first_tau_factor = 0.5*(max(abs(A'*b))/tau);
%steps = 5;%5
%debias = 0;
%stopCri=5;
%tolA=1.e-5;
%fprintf('=========== KCS ============\n');
%[x_rec,x_debias,objective,times,debias_start,mses,taus]= ...
%    GPSR_BB(b,A,AT,tau,...
%          'Debias',debias,...
%          'Monotone',0,...
%          'Initialization',2,...
%          'MaxiterA',100,...
%          'True_x',x(:),...
%          'StopCriterion',stopCri,...
%       	  'ToleranceA',tolA,...
%          'Verbose',0);
%xrec_kcs = reshape(x_rec,[w,h,s]);
%i=5;[mp(i),sm(i),er(i),sa(i)] = msqia(xrec_kcs/255,x/255);

