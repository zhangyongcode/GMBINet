% This code is used to calculate the PFOM (Pratt’s figure of merit) metric.
% If you use this code in your paper, please cite "Quantitative design and evaluation of enhancement/thresholding edge detectors".

close all;
clear;
clc;
addpath(genpath('.'));              

Gt_path='F:\ZhangyongDoctor\Surf\SD-saliency\SD-saliency-900\ground-truth\';      % ground truth 
Salcut_path='F:\ZhangyongDoctor\Surf\SD-saliency\prediction-others\MINetPaperResults\DACNet\';       % saliency detection results 
Gt_list=dir(strcat(Gt_path,'*.png'));             
Salcut_list=dir(strcat(Salcut_path,'*.png'));     
imgNum=length(Salcut_list);                       



if imgNum~=length(Gt_list)
    error('the number of images must be the same!');
end
%--------------------------------------------------%
PROM = zeros(1,imgNum);
parfor i=1:imgNum
    GT = imread(strcat(Gt_path,Gt_list(i).name));     % GT [0, 255]
%     GT = im2double(imgGT_gray);
    %disp(Gt_list(i).name);
    Sal = imread(strcat(Salcut_path,Salcut_list(i).name));  
    if size(Sal,3) > 1
        Sal = Sal(:,:,1);
    end
    [Ea_gt, Ed_sal] = SalToEdge(GT, Sal);       
    [PROM(i),~,~] = pratt(Ea_gt, Ed_sal);    % Ea_gt, Ed_sal is single channel image ranging from [0, 1]
    
end
prom = mean2(PROM);
%%
disp('显示计算结果');
disp(Salcut_path);
disp(['PROM=',num2str(prom/100)]);








