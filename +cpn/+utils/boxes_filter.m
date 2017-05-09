function [aboxes, idx] = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, nms_score_thresh)
% Inputs:
%     aboxes : [x1,y1,x2,y2, score]
% Outputs:
%     aboxes : boxes after nms
%     idx: indices of retained boxes
% 

if nargin < 6
    nms_score_thresh = 0;
end
if nargin < 5
    use_gpu = gpuDeviceCount>0;
end
if nargout == 1
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(size(aboxes,1), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(cpn.utils.nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(size(aboxes,1), after_nms_topN), :);
    end
    
    idx = [];
else
    % to speed up nms
    if per_nms_topN > 0
        idx = [1:min(size(aboxes,1), per_nms_topN)]';
        aboxes = aboxes(idx, :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        idx2 = cpn.utils.nms(aboxes, nms_overlap_thres, use_gpu);
        idx = idx(idx2,1);
        aboxes = aboxes(idx2, :);  
    end
    if after_nms_topN > 0
        idx3 = 1:min(size(aboxes,1), after_nms_topN);
        idx = idx(idx3,1);
        aboxes = aboxes(idx3, :);
    end
end
if nms_score_thresh > 0
    idx_del = aboxes(:,5) < nms_score_thresh;
    aboxes(idx_del, :) = [];
    idx(idx_del) = [];
end
end
