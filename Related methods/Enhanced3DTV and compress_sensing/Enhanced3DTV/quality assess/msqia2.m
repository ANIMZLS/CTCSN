function [mpsnr, mssim, ergas, sam, rmse] = msqia2(imagery1, imagery2)
    % Evaluates the quality assessment indices for two HSIs.

    [M, N, p] = size(imagery1);
    psnr_vector = zeros(1, p);
    ssim_vector = zeros(1, p);
    sam_vector = zeros(1, p);
    rmse_vector = zeros(1, p);

    for i = 1:p
        J = imagery1(:, :, i);
        I = imagery2(:, :, i);

        % PSNR
        psnr_vector(i) = PSNR_c(J,I,M,N);

        % SSIM
        [ssim_vector(i), ~] = ssim(J,I);

        % SAM
        ref_band = J(:);
        target_band = I(:);
        dot_product = sum(ref_band .* target_band);
        norm_ref = norm(ref_band);
        norm_target = norm(target_band);
        sam_vector(i) = acosd(dot_product / (norm_ref * norm_target));

        % RMSE
        squared_diff = (ref_band - target_band).^2;
        aux = sum(sum(squared_diff, 1), 2) / (size(ref_band, 1) * size(ref_band, 2));
        rmse_vector(i) = sqrt(sum(aux));
    end

    % Calculate mean values
    mpsnr = mean(psnr_vector);
    mssim = mean(ssim_vector);
    sam = mean(sam_vector);
    rmse = mean(rmse_vector);
    ergas = ErrRelGlobAdimSyn(imagery1, imagery2);

    % Print or display the results
    fprintf('PSNR: %.4f\n', mpsnr);
    fprintf('SSIM: %.4f\n', mssim);
    fprintf('SAM: %.4f\n', sam);
    fprintf('RMSE: %.4f\n', rmse);
    fprintf('ERGAS: %.4f\n', ergas);
    
end
