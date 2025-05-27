
%weighted F-measure (WF)
gt_dir = 'F:\ZhangyongDoctor\Surf\SD-saliency\SD-saliency-900\ground-truth\'; % 预测结果
pre_dir = 'F:\ZhangyongDoctor\Surf\SD-saliency\prediction-others\MINetPaperResults\DACNet\'; % 训练标签
gt_files = dir(strcat(gt_dir,'*.png'));
pre_files = dir(strcat(pre_dir,'*.png'));
Beta2=1;
WF = zeros(length(gt_files), 1);
for k = 1:length(gt_files)
    PreImg =  imread(strcat(pre_dir,pre_files(k).name));   
    
    gtImg = imread(strcat(gt_dir,gt_files(k).name));
    
    WF(k) = WFb(PreImg, gtImg, Beta2);
end

wf = mean(WF);
fprintf('WF for %f\n', wf);