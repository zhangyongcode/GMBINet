
gt_dir = 'F:\ZhangyongDoctor\Surf\SD-saliency\SD-saliency-900\ground-truth\'; % 预测结果
pre_dir = 'F:\ZhangyongDoctor\Surf\SD-saliency\prediction-others\MINetPaperResults\DACNet\'; % 训练标签
gt_files = dir(strcat(gt_dir,'*.png'));
pre_files = dir(strcat(pre_dir,'*.png'));

MAE = zeros(length(gt_files), 1);
for k = 1:length(gt_files)
    PreImg =  imread(strcat(pre_dir,pre_files(k).name));
    
    
    gtImg = imread(strcat(gt_dir,gt_files(k).name));
    
    MAE(k) = MAE2(PreImg, gtImg);
end

mae = mean(MAE);
fprintf('MAE for %f\n', mae);