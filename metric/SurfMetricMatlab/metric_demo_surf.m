% zhangyong7630
clc;clear

gt_dir = ''; %  标签
pre_dir = '';% 预测结果
%MAE 
gt_files = dir(strcat(gt_dir,'*.png'));
pre_files = dir(strcat(pre_dir,'*.png'));

MAE = zeros(length(gt_files), 1);
for k = 1:length(gt_files)
    PreImg =  imread(strcat(pre_dir,pre_files(k).name));   
    gtImg = imread(strcat(gt_dir,gt_files(k).name));
    
    MAE(k) = MAE2(PreImg, gtImg);
end
mae = mean(MAE);
% WF 
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
% OR 
gt_files = dir(strcat(gt_dir,'*.png'));
pre_files = dir(strcat(pre_dir,'*.png'));
setCurve = false;
OR = zeros(length(gt_files), 1);
for k = 1:length(gt_files)
    PreImg =  imread(strcat(pre_dir,pre_files(k).name));   
    
    gtImg = imread(strcat(gt_dir,gt_files(k).name));
    
    OR(k) = CalOverlapRatio(PreImg, gtImg, setCurve);
end

or = mean(OR);
% -----------------------SM---------------------
gtPath = gt_dir;

fgPath = pre_dir;

% load the gtFiles
gtFiles = dir(fullfile(gtPath,'*.png'));

% for each gtFiles
S_score = zeros(1,length(gtFiles));
for i = 1:length(gtFiles)
    % fprintf('Processing %d/%d...\n',i,length(gtFiles));
    
    % load GT
    [GT,map] = imread(fullfile(gtPath,gtFiles(i).name));
    if numel(size(GT))>2
        GT = rgb2gray(GT);
    end
    GT = logical(GT);
    
    % in some dataset(ECSSD) some ground truth is reverse when map is not none
%     if ~isempty(map) && (map(1)>map(2))
%         GT = ~GT;
%     end
    
    % load FG
    prediction = imread(fullfile(fgPath,gtFiles(i).name));
    if numel(size(prediction))>2
        prediction = rgb2gray(prediction);
    end
    
    % Normalize the prediction.
    d_prediction = double(prediction);
    if (max(max(d_prediction))==255)
        d_prediction = d_prediction./255;
    end
    d_prediction = reshape(mapminmax(d_prediction(:)',0,1),size(d_prediction));
    
    % evaluate the S-measure score
    score = StructureMeasure(d_prediction,GT);
    S_score(i) = score;
    
end
% -----------------------------PFOM-------------------------
Gt_path=gt_dir;      % ground truth 
Salcut_path=pre_dir;       % saliency detection results 
Gt_list=dir(strcat(Gt_path,'*.png'));             
Salcut_list=dir(strcat(Salcut_path,'*.png'));     
imgNum=length(Salcut_list);                       



if imgNum~=length(Gt_list)
    error('the number of images must be the same!');
end
%--------------------------------------------------%

error_number= 0;
PROM = zeros(1,imgNum);
parfor i=1:imgNum
    GT = imread(strcat(Gt_path,Gt_list(i).name));     % GT [0, 255]
%     GT = im2double(imgGT_gray);
    %disp(Gt_list(i).name);
    Sal = imread(strcat(Salcut_path,Gt_list(i).name));  


    if size(Sal,3) > 1
        Sal = Sal(:,:,1);
    end
    if size(GT,3) > 1
        GT = GT(:,:,1);
    end
    try
        [Ea_gt, Ed_sal] = SalToEdge(GT, Sal); 

        [m,n] = size(Ea_gt)
        [PROM(i),~,~] = pratt(Ea_gt, Ed_sal);    % Ea_gt, Ed_sal is single channel image ranging from [0, 1]
    catch
        % disp(Salcut_list(i).name)
        error_number = error_number + 1;
    end
        
end

disp(pre_dir)
disp(error_number)

fprintf('MAE:%.4f, WF:%.4f, OR:%.4f SM:%.4f PFOM:%.4f\n', mae, wf, or, mean2(S_score),mean2(PROM)/100);
