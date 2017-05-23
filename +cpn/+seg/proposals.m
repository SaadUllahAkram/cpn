function [stats, score_cell, mask_cell, rois_cell, idx] = proposals(conf, seg_net, im_cell, rois_cell)
% Returns the segmentation for given rois
%
% Inputs:
%     im_cell: cell array of images
%     rois_cell: cell array of [xmin ymin xmax ymax prob.]
% Outputs:
%     stats: seg proposal stats
%     score: prob.
%     mask : labelled mask
%



num_rois = bia.caffe.get_max_rois(seg_net);
if strcmp(conf.dataset, 'Hist-BM')
    num_rois = 1000;%
end
db = conf.debug;

if length(im_cell) < 4
    verbose         = 0;
else
    verbose         = 1;
end
thresh          = 0.5;
mask_type       = 3;

if isnumeric(im_cell) && isnumeric(rois_cell)
    error('im: should be in a cell array')
end

N           = length(im_cell);% num of images to evaluate
stats       = cell(N,1);
mask_cell   = cell(N,1);
score_cell  = cell(N,1);
if verbose
    tic_start = tic;
    fprintf('%d:  ', N)
end
for i=1:N
    if verbose
        fprintf(repmat('\b',1, sum([ (i-1) >0 (i-1)>9 (i-1)>99])))
        fprintf('%d', i)
    end
    im      = im_cell{i};
    rois    = rois_cell{i};
    sz      = [size(im, 1), size(im, 2)];
    
    if isempty(rois)
        tmp            = struct('PixelIdxList',{},'Area',{},'BoundingBox',{},'Centroid',{},'Score',{});
        stats{i}       = tmp;
        score_cell{i}  = zeros(sz);
        mask_cell{i}   = zeros(sz);
        continue
    end
    R       = size(rois,1);% # of rois
    rois(:,1:4) = bia.convert.bb(rois(:,1:4), 'b2c');%bb_2corners
    
    sz_orig = [size(im, 1), size(im, 2)];
    rois(:, 1:4) = cpn.seg.adjust_rois(conf, rois(:, 1:4), size(im));
    conf.feats = 'fc1';
    if isfield(conf, 'roi_old') && conf.roi_old
        rois2 = rois;
        rois2(:, 1:4) = cpn.seg.adjust_rois(conf, rois2(:, 1:4), size(im));
    else
        rois2 = rois;
    end
    [rois_score_vec, rois_prob, feats] = cspn_im_detect(conf, seg_net, im, rois2(:, 1:4), num_rois);
    
    seg_vec_size = sqrt(length(rois_score_vec(1,:)));
    
    if isnan(rois_score_vec(:))
        fprintf('NaN in segmentation masks: %% of pixels: %1.2f\n', sum(isnan(rois_score_vec(:)))/numel(rois_score_vec))
    end
    
    score   = zeros(sz);% prob of a pixel being cell
    mask    = zeros(sz);% labelled seg image
    
    % do NMS
    rect_crop         = round(rois(:,[2 4 1 3]));% rois=[xmin ymin xmax ymax] -> rect_crop=[ymin ymax xmin xmax]
    rect_crop(:,[1,3])= max(1, rect_crop(:,[1,3]));
    rect_crop(:,[2,4])= min(repmat([size(score, 1), size(score, 2)], size(rect_crop, 1), 1), rect_crop(:,[2,4]));
    
    px_cell     = cell(R,1);
    px_bw_cell  = cell(R,1);
    clear   stats_loc
    
    if ~isempty(feats)
        func = @(a,b,c,d,e,f) cpn_roi_stats(a, b, c, d, e, f);
    else
        feats = zeros(R,1);
        func = @(a,b,c,d,e,f) cpn_roi_stats(a, b, c, d, e);
    end
    
    parfor j = 1:R% to speed up imresize
        px  = reshape(rois_score_vec(j, :), seg_vec_size, seg_vec_size)';% transpose
        r   = rect_crop(j, :);
        bsz = [r(2)-r(1)+1, r(4)-r(3)+1];
        % px  = imresize(px, bsz, 'nearest');
        px  = imresize(px, bsz, 'bicubic');
        px_cell{j} = px;
        %             px_bw   = (px > thresh)
        px_bw   = keep_largest_region(px > thresh);
        px_bw_cell{j} = px_bw;
        stats_loc(j,1) = func(px_bw, r, sz_orig, 0, rois(j,5), feats(j,:));
    end
    stats{i} = stats_loc;
    %imshow(bia.draw.boundary([],im_cell{i}, stats{i}))
    %bia.plot.centroids('',stats{i});
    for j = 1:R
        r   = rect_crop(j, :);
        tmp_lab = mask(r(1):r(2), r(3):r(4));
        px  = px_cell{j};
        px_bw = px_bw_cell{j};
        %     px  = imerode(px, ones(10));
        score(r(1):r(2), r(3):r(4)) = max(px, score(r(1):r(2), r(3):r(4)));% keep max prob. for each pixel
        if mask_type == 3
            px_bw = j*px_bw;
        end
        if mask_type == 2 || mask_type == 3% ismember can take few sec.
            tmp_lab(tmp_lab==0) = px_bw(tmp_lab==0);% only update unassigned pixels
        elseif mask_type == 1
            tmp_lab = max(px_bw, tmp_lab);% keep max prob. for each pixel
        end
        mask(r(1):r(2), r(3):r(4)) = tmp_lab;
    end
    
    mask_cell{i}  = mask;
    score_cell{i} = score;
    
    if db
        imshow(bia.draw.boundary('',im,stats{i}([stats{i}.Score]>0.98,1)))
        drawnow
    end
end
for i=1:N
    [stats{i}, idx{i,1}] = bia.struct.standardize(stats{i},'seg');
    rois_cell{i} = rois_cell{i}(idx{i,1}, :);
end
if verbose
    % count_chars = N + sum([ (1:N) >0 (1:N)>9 (1:N)>99]);% counts # of chars printed in loop above
    count_chars = sum([ N>0 N>9 N>99]);
    fprintf(repmat('\b',1,count_chars))
    fprintf('Time (per image): %1.1f sec\n', toc(tic_start)/N)
end
if N == 0
    stats        = stats{1};
    mask_cell    = mask_cell{1};
    score_cell   = score_cell{1};
end
end


function stats = cpn_roi_stats(mask, roi, sz, pad, score, feats)
% rois: []
% mask: binary mask
% sz: size of full image
% stats:

idx     = find(mask);
[r,c]   = ind2sub(size(mask), idx);
r       = double(r+roi(1)-1-pad);
c       = double(c+roi(3)-1-pad);
idx_rm  = r < 1 | c < 1 | r > sz(1) | c > sz(2);
r(idx_rm) = [];
c(idx_rm) = [];

stats.PixelIdxList = sub2ind(sz, r, c);
stats.Area = length(stats.PixelIdxList);

% code snippets copied from "regionprops"
num_dims = numel(sz);
list = [c, r];
if (isempty(list))
    stats.BoundingBox = [0.5*ones(1,num_dims) zeros(1,num_dims)];
else
    min_corner = min(list,[],1) - 0.5;
    max_corner = max(list,[],1) + 0.5;
    stats.BoundingBox = [min_corner (max_corner - min_corner)];
end
stats.Centroid = mean([c,r],1);
stats.Score = score;
if nargin >= 6
    stats.Features = feats;
end
end


function bw = keep_largest_region(bw)
stats = regionprops(logical(bw), 'Area', 'PixelIdxList');
areas = [stats(:).Area];
[~, idx] = max(areas);
for i=1:length(stats)
    if i~= idx
        bw(stats(i).PixelIdxList) = 0;
    end
end
bw = imfill(bw, 'holes');
end


function [pred_masks, scores, feats] = cspn_im_detect(conf, caffe_net, im, boxes, max_rois_num_in_gpu)
% pred_masks --> numROIS x 144, when doing segmentation

im_size = size(im);
conf.channels = size(caffe_net.blobs('data').get_data(), 3);
if isfield(conf, 'seg')
    conf = bia.utils.setfields(conf, 'test_scales', min(im_size(1:2)), 'test_max_size', max(im_size(1:2)));
end
[im_blob, rois_blob, ~] = get_blobs(conf, im, boxes);

% When mapping from image ROIs to feature map ROIs, there's some aliasing
% (some distinct image ROIs get mapped to the same feature ROI).
% Here, we identify duplicate feature ROIs, so we only compute features
% on the unique subset.
[~, index, inv_index] = unique(rois_blob, 'rows');
rois_blob = rois_blob(index, :);
boxes = boxes(index, :);

% permute data into caffe c++ memory, thus [num, channels, height, width]
if length(size(im_blob)) == 3%rgb image
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
end
im_blob = permute(im_blob, [2, 1, 3, 4]);
im_blob = single(im_blob);
rois_blob = rois_blob - 1; % to c's index (start from 0)
rois_blob = permute(rois_blob, [3, 4, 2, 1]);
rois_blob = single(rois_blob);

total_rois = size(rois_blob, 4);
total_seg_masks = cell(ceil(total_rois / max_rois_num_in_gpu), 1);
for i = 1:ceil(total_rois / max_rois_num_in_gpu)
    sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
    sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
    sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
    
    net_inputs = {im_blob, sub_rois_blob};
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs);
    
    % use softmax estimated probabilities
    if ismember(conf.mode, 1)
        seg_masks = output_blobs{1};% 144 x 2 x numROIS
        seg_masks = squeeze(seg_masks);
        if length(size(seg_masks)) == 2% entropy-loss
            seg_masks = permute(seg_masks, [2 1]);% numROIS x 144 x 2
        else
            seg_masks = permute(seg_masks, [3 1 2]);% numROIS x 144 x 2
        end
        total_seg_masks{i} = seg_masks;
    end
    if isfield(conf, 'feats') && ismember(conf.feats, caffe_net.blob_names)
        feats{i,1} = caffe_net.blobs(conf.feats).get_data()';
    else
        feats{i,1} = [];
    end
end

feats = cell2mat(feats);
if ismember(conf.mode, 1)
    seg_masks = cell2mat(total_seg_masks);
    if length(size(seg_masks)) == 2
        pred_masks = seg_masks(inv_index, :);% numROIS x 144
    else
        pred_masks = seg_masks(inv_index, :, 2:end);% numROIS x 144
    end
end
% Map scores and predictions back to the original set of boxes
scores = ones(size(pred_masks, 1),1);
if ~isempty(feats)
    feats = feats(inv_index, :);
end
if isfield(conf, 'feats') && ismember(conf.feats, caffe_net.blob_names)
    assert(size(feats, 1) == size(scores, 1))
end
end


function [data_blob, rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
if conf.channels == 3 && size(im, 3) == 1
    im = repmat(im, [1 1 3]);
end
im_scale_factors = 1;
data_blob = single(im) - conf.image_means;
rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end


function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
[feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
rois_blob = single([levels, feat_rois]);
end


function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
im_rois = single(im_rois);

if length(scales) > 1
    widths = im_rois(:, 3) - im_rois(:, 1) + 1;
    heights = im_rois(:, 4) - im_rois(:, 2) + 1;
    
    areas = widths .* heights;
    scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
    [~, levels] = min(abs(scaled_areas - 224.^2), [], 2);
else
    levels = ones(size(im_rois, 1), 1);
end

feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end


function boxes = clip_boxes(boxes, im_width, im_height)
% x1 >= 1 & <= im_width
boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
% y1 >= 1 & <= im_height
boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
% x2 >= 1 & <= im_width
boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
% y2 >= 1 & <= im_height
boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end