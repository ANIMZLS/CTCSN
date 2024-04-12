function [mpsnr,mssim,ergas,sam,rmse]=msqia(imagery1, imagery2)
% Evaluates the quality assessment indices for two HSIs.
%
% Syntax:
%   [mpsnr, mssim,ergas ] = MSIQA(imagery1, imagery2)
% Input:
%   imagery1 - the reference HSI data array
%   imagery2 - the target HSI data array
%   NOTE: MSI data array  is a M*N*K array for imagery with M*N spatial
%	pixels, K bands and DYNAMIC RANGE [0,1];
[M,N,p]  = size(imagery1);
psnrvector=zeros(1,p);
for i=1:1:p
    J=imagery1(:,:,i);
    I=imagery2(:,:,i);
    psnrvector(i)=PSNR_c(J,I,M,N);
end 
mpsnr = mean(psnrvector);

SSIMvector=zeros(1,p);
for i=1:1:p
    J=imagery1(:,:,i);
    I=imagery2(:,:,i); 
    [ SSIMvector(i),~] = ssim(J,I);
end
mssim=mean(SSIMvector);

sam_vector = zeros(1, p);
for i = 1:p
    ref_band = imagery1(:, :, i);
    target_band = imagery2(:, :, i);

    % Compute dot product and norms without normalization
    dot_product = sum(ref_band(:) .* target_band(:));
    norm_ref = norm(ref_band(:));
    norm_target = norm(target_band(:));

    % Compute SAM
    sam_vector(i) = acosd(dot_product / (norm_ref * norm_target));
end
sam = mean(sam_vector);

rmse_vector = zeros(1, p);
for i = 1:p
    ref_band = imagery1(:, :, i);
    target_band = imagery2(:, :, i);

    rmse_vector(i) = sqrt(sum((ref_band(:) - target_band(:)).^2) / numel(ref_band));
end
    rmse = mean(rmse_vector);

ergas = ErrRelGlobAdimSyn(imagery1, imagery2);