% Step 1: ozone.data.r loads the data, standardizes it (so each variable has mean 0 and variance 1), and saves it for matlab
% --> Step 2: ozone.m pre-whitens each variable
% Step 3: ozone.r runs the PC algorithm on the original data and the pre-whitened data and makes plots
addpath(genpath('D:\Dropbox\Academics\My Papers\Causality\GPIT\tist-supporting-materials\'));
addpath(genpath('D:\Dropbox\Academics\My Papers\Causality\GPIT\tist-supporting-materials\'));
addpath(genpath('D:\Dropbox\Academics\My Papers\Causality\GPIT\tist-supporting-materials\HSIC\MATLAB_version\'));
%%
[totale_positivi,txt,raw] = xlsread('totale_positivi.xlsx');
totale_positivi = totale_positivi(end-50:end,:);
csvwrite('totale_positivi.csv', totale_positivi);
dates = txt(2:end,1); mat = txt(1,2:end); mat = mat'; 
for j = 1:length(mat); names{j,1} = strrep(mat{j},'totale_positivi_',''); end

x = [1:size(totale_positivi,1)]'; [n, nin] = size(x);

c=1;
for i = 1:size(totale_positivi,2)
    i
  y = totale_positivi(:,i); rawX(:,c) = y; c=c+1;
  lik = lik_gaussian('sigma2', 0.2^2,'sigma2_prior', prior_logunif());
  gpcf = gpcf_exp('lengthScale', 1.1, 'magnSigma2', 0.2^2,...
    'lengthScale_prior', prior_unif(), 'magnSigma2_prior', prior_sqrtunif()); 
  
  gp = gp_set('lik', lik, 'cf', gpcf);
  opt=optimset('TolFun',1e-3,'TolX',1e-3);
  gp=gp_optim(gp,x,y,'opt',opt);

  [w,s]=gp_pak(gp);  
  [Eft_map, Varft_map] = gp_pred(gp, x, y, x);
  
  whitened(:,i) = y - Eft_map;
end 

csvwrite('totale_positivi_prewhitened.csv', whitened)
%% Plots
subplot(2,1,1); plot(rawX); 
subplot(2,1,2); plot(whitened);


