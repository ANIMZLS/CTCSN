clc; 
clear; close all

Xim1 = multibandread('f060810t01p00r08rdn_c_sc01_ort_img', [5336,1000,224], 'int16', 0, 'bip', 'ieee-be',...
{'Row','Range',[800 4500]}, {'Column','Range',[250 850]});

Xim1(:,:,215:224) = [];
Xim1(:,:,152:170) = [];
Xim1(:,:,104:116) = [];
Xim1(:,:,1:10) = [];
% AVIRIS remove 1-10, 104-116, 152-170, and 215-224  

Xim1( Xim1 < 0 ) = 0;

band_set=[25 15 6]; % AVIRIS RGB bands  
normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;


x = 1;

a=size(Xim1,1);
b=size(Xim1,2);
a=fix(a/256);
b=fix(b/256);

for i=1:a
   for j=1:b
       Xim=Xim1((i-1)*256+1:i*256,(j-1)*256+1:j*256,:);
       save(['HSI/data_' num2str(x) '.mat'],'Xim');
       temp_show=Xim(:,:,band_set);
       temp_show=normColor(temp_show);
       imwrite(temp_show, "HSI_figs/data_"+ num2str(x)+ ".jpg", 'jpg');
       x=x+1;
   end    
end


figure
temp_show_wholefig=Xim1(:,:,band_set);
temp_show_wholefig=normColor(temp_show_wholefig);
imshow(temp_show_wholefig);

