function [mpsnr, mssim, ergas, msam] = msqia1(imagery1, imagery2)
    [M, N, p] = size(imagery1);
    
    % ����ÿ�����ε� PSNR��SSIM �� SAM
    psnrvector = zeros(1, p);
    ssimvector = zeros(1, p);
    samvector = zeros(1, p);

    for i = 1:p
        J = 255 * imagery1(:, :, i);
        I = 255 * imagery2(:, :, i);

        % ���� PSNR
        psnrvector(i) = PSNR_c(J, I, M, N);

        % ���� SSIM
        [ssimvector(i), ~] = ssim(J, I);

        % ���� SAM
        samvector(i) = SAM(imagery1(:, :, i), imagery2(:, :, i));
    end

    % ����ƽ��ֵ
    mpsnr = mean(psnrvector);
    mssim = mean(ssimvector);
    ergas = ErrRelGlobAdimSyn(255 * imagery1, 255 * imagery2);
    msam = mean(samvector);
end
