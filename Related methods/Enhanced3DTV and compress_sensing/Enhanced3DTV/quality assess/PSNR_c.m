function result = PSNR_c(ReferBuffer, UnReferBuffer, lHeight, lWidth)

    result = 0;
    psnrMax = max(UnReferBuffer(:));  % maxval

    for j = 1:lWidth * lHeight
        temp = double(ReferBuffer(j) - UnReferBuffer(j));
        result = result + temp * temp;
    end
    
    if (result == 0)
        result = 100;
    else
        mse = result / (lWidth * lHeight);
        result = 10 * log10(psnrMax^2 / mse);
    end
end
