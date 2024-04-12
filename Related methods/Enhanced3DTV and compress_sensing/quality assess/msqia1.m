function [mpsnr, mssim, ergas, msam] = msqia1(imagery1, imagery2)
    [M, N, p] = size(imagery1);
    
    % 计算每个波段的 PSNR、SSIM 和 SAM
    psnrvector = zeros(1, p);
    ssimvector = zeros(1, p);
    samvector = zeros(1, p);

    for i = 1:p
        J = 255 * imagery1(:, :, i);
        I = 255 * imagery2(:, :, i);

        % 计算 PSNR
        psnrvector(i) = PSNR_c(J, I, M, N);

        % 计算 SSIM
        [ssimvector(i), ~] = ssim(J, I);

        % 计算 SAM
        samvector(i) = SAM(imagery1(:, :, i), imagery2(:, :, i));
    end

    % 计算平均值
    mpsnr = mean(psnrvector);
    mssim = mean(ssimvector);
    ergas = ErrRelGlobAdimSyn(255 * imagery1, 255 * imagery2);
    msam = mean(samvector);
end
