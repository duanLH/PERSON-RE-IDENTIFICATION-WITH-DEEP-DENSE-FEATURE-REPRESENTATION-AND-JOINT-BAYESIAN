clc;clear all;close all;
%***********************************************%
% This code runs on the Market-1501 dataset.    %
% Please modify the path to your own folder.    %
% We use the mAP and hit-1 rate as evaluation   %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

% Please download Market-1501 dataset and unzip it in the "dataset" folder.



%% l the ID and camera for database images
testID = load('data/testID');
testID=testID.testID;
testCAM = load('data/testCAM');
testCAM=testCAM.testCAM;
[nTest,m]=size(testID);


%% load the ID and camera for query images
queryID = load('data/queryID');
queryID=queryID.queryID;
queryCAM = load('data/queryCAM');
queryCAM=queryCAM.queryCAM;
[nQuery,m] = size(queryID);
 



%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision

ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query

r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)


dist_euclidean=readNPY('D_Euclidean.npy');
dist_euclidean=dist_euclidean(1:19732,1:3368);
dist_jointbayesian=readNPY('D_JointBayesian.npy');
dist_jointbayesian=dist_jointbayesian(1:19732,1:3368);

knn = 1; % number of expanded queries. knn = 1 yields best result



fprintf ('dealing...\n');
for k = 1:nQuery
    if rem(k,500)==0
       fprintf('dealing %d query image\n',k);
    end
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    tic
    score_euclidean = dist_euclidean(:, k);
    score_jointbayesian = dist_jointbayesian(:, k);
    
    % sort database images according Euclidean distance
    [~, index_euclidean] = sort(score_euclidean, 'ascend');  % single query
    [~, index_jointbayesian] = sort(score_jointbayesian, 'ascend'); 
    
    [ap_euclidean(k), CMC_euclidean(k, :)] = compute_AP(good_index, junk_index, index_euclidean);% compute AP for single query
    [ap_jointbayesian(k), CMC_jointbayesian(k, :)] = compute_AP(good_index, junk_index, index_jointbayesian);% compute AP for single query
    
end

CMC_euclidean = mean(CMC_euclidean);
CMC_jointbayesian = mean(CMC_jointbayesian);
save 'result.mat' CMC ap;
%% print result
fprintf('single query: Euclidean:        mAP = %f, rank1,5,10,20 precision = %f %f %f %f\r\n', mean(ap_euclidean), CMC_euclidean(1),CMC_euclidean(5),CMC_euclidean(10),CMC_euclidean(20));
fprintf('single query: JointBayesian:    mAP = %f, rank1,5,10,20 precision = %f %f %f %f\r\n', mean(ap_jointbayesian), CMC_jointbayesian(1),CMC_jointbayesian(5),CMC_jointbayesian(10),CMC_jointbayesian(20));





