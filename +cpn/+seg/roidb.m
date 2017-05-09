function [image_roidb, bbox_means, bbox_stds] = roidb(conf, roidbs, bbox_means, bbox_stds)

if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end

if ~iscell(roidbs)
    roidbs = {roidbs};
end

roidbs = roidbs(:);

image_roidb = ...
    cellfun(@(y) ... // @(imdbs, roidbs)
    arrayfun(@(z) ... //@([1:length(x.image_ids)])
    struct('image_id', y(z).image_id, 'im_size', y(z).sz, ...
    'overlap', y(z).overlap, 'boxes', y(z).boxes, 'class', y(z).class, 'image', [], 'bbox_targets', []), ...
    [1:length(y)]', 'UniformOutput', true),...
    roidbs, 'UniformOutput', false);

image_roidb = cat(1, image_roidb{:});

% enhance roidb to contain bounding-box regression targets
[image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end


function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
% means and stds -- (k+1) * 4, include background class
db_show = 0;
num_images = length(image_roidb);
valid_imgs = true(num_images, 1);

disp_iter = max(1, round(num_images/20));
fprintf('%d::', num_images)
time_str = '';
time_start = tic;

image_roidb(1).overlap_bb = [];
image_roidb(1).labels = [];
image_roidb(1).rois = [];
for i = 1:num_images
    if rem(i, disp_iter) == 0
        fprintf(repmat('\b', 1, length(time_str)))
        time_str = sprintf('%3.0f%%: %4.1fsec', 100*i/num_images, toc(time_start));
        fprintf('%s', time_str)
    end
    
    if ismember(conf.pos_criteria, [2,3,5])
        %image_roidb(i) = compute_seg_overlap(conf, image_roidb(i));%not being used
    end
    
    rois = image_roidb(i).boxes;
    boxes_orig = image_roidb(i).boxes;
    [image_roidb(i).bbox_targets, valid_imgs(i), idx_train] = compute_targets(conf, rois, image_roidb(i));
    if 1% only use boxes which satisfy the criteria in 'compute_targets' for training
        image_roidb(i).overlap = image_roidb(i).overlap(idx_train, :);
        image_roidb(i).boxes = image_roidb(i).boxes(idx_train, :);
        image_roidb(i).class = image_roidb(i).class(idx_train, :);
        image_roidb(i).bbox_targets = image_roidb(i).bbox_targets(idx_train, :);
        rois = rois(idx_train, :);
        boxes_orig = boxes_orig(idx_train, :);
    end

    rois(:, 1:4) = cpn.seg.adjust_rois(conf, rois(:, 1:4), image_roidb(i).im_size);
    
    if ismember(conf.mode, 1)% seg_masks : 1(central cell), 0(BG), 2(other cells)
        [image_roidb(i).seg_masks, image_roidb(i).seg_weights, image_roidb(i).markers, image_roidb(i).stats] = ...
            compute_seg_targets(conf, image_roidb(i).image_id, rois);
    end
    image_roidb(i) = add_fg_bg_labels(conf, rois, image_roidb(i));
    image_roidb(i).rois = rois;
%     image_roidb(i).boxes = rois;
    if db_show
        figure(1)
        im = conf.imread(image_roidb(i).image_id);
        idx = 1:ceil(size(rois,1)/50):size(rois,1);
        rr = boxes_orig(idx, :);
        tar = image_roidb(i).bbox_targets(idx, 2:end);
        stats = image_roidb(i).stats(idx);
        im2 = bia.draw.boundary([], im, stats);
        imshow(im2)
        bb = cpn.utils.fast_rcnn_bbox_transform_inv(rr, tar);% gt
        bia.plot.bb([],bia.convert.bb(rr,'c2m'))
        bia.plot.bb([],bia.convert.bb(bb,'c2m'), 'g')
        drawnow
    end
end
fprintf('\n')
if ~all(valid_imgs)
    image_roidb = image_roidb(valid_imgs);
    fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
end
end


function roidb = add_fg_bg_labels(conf, rois, roidb)
overlaps = max(roidb.overlap, [], 2);

N = size(rois,1);
labels = zeros(N,1);
% Select foreground ROIs as those with >= FG_THRESH overlap
if ismember(conf.pos_criteria, [1 2 3 4 5 13])
    markers = roidb.markers;
    in = zeros(N,1);
    for i=1:N
        r = rois(i,:);
        in(i,1) = sum(r(1)<markers(:,1) & r(3)>markers(:,1) & r(2)<markers(:,2) & r(4)>markers(:,2));
    end
end

if conf.pos_criteria == 1% use only markers
    fg_inds = find(in == 1);
    bg_inds = find(in ~= 1 & overlaps >= conf.bg_thresh_lo);
elseif conf.pos_criteria == 2% only use IoU
    fg_inds = find(overlaps >= conf.fg_thresh);
    bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo);
elseif ismember(conf.pos_criteria, [3,13])%active: 24feb-2017 % 13: makes bg regions have 0 mask.
    fg_inds = find(overlaps >= conf.fg_thresh & in == 1);% use both IoU and markers
    bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo & in ~= 1);
elseif conf.pos_criteria == 4
    fg_inds = find(overlaps >= conf.fg_thresh & in == 1);% use both IoU and markers
    bg_inds = find(in ~= 1 & overlaps >= conf.bg_thresh_lo);
elseif conf.pos_criteria == 5% only % of mask pixels
    fg_inds = find( sum(roidb.seg_masks>0,2)./size(roidb.seg_masks,2) > 0.3);
    bg_inds = find( sum(roidb.seg_masks>0,2)./size(roidb.seg_masks,2) < 0.3 );
else
    fg_inds = find(overlaps >= conf.fg_thresh);
    bg_inds = [];
end
% % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
% if ismember(conf.mode, 11)
%     bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo & ~roidb.seg_valid);
% end
if conf.ignore_invalid% get rid of invalid entries
    fg_inds(roidb.bbox_targets(fg_inds, 1) == -1) = [];
    bg_inds(roidb.bbox_targets(bg_inds, 1) == -1) = [];
end
labels(fg_inds) = 1;
labels(bg_inds) = -1;

roidb.labels = labels;
end


function a = compute_seg_overlap(conf, a)
class = find(a.overlap(1,:));
idx = setdiff(1:size(a.overlap,2), class);
assert(bia.utils.ssum(a.overlap(:,idx)) == 0)

a.overlap_bb = a.overlap;
a.overlap = 0*a.overlap;

[~,~,mask] = conf.gtread(a.image_id);

b = bia.convert.bb(a.boxes, 'c2b');
area = b(:,3).*b(:,4);
b = bia.convert.bb(b,'b2r');
for i=1:size(b,1)
    if a.overlap_bb(i,class) == 1
        a.overlap(i,class) = 1;
    else
        % find the largest cell and its size
        im = mask(b(i,1):b(i,2), b(i,3):b(i,4));
        tmp = regionprops(im, 'Area');
        if isempty(tmp)
            max_size = 0;
        else
            max_size = max([tmp(:).Area]);
        end
        a.overlap(i,class) = max_size/area(i);
    end
end
end


function [bbox_targets, is_valid, idx] = compute_targets(conf, rois, image_roidb)
class = unique(image_roidb.class);
max_overlaps = full(image_roidb.overlap(:, class));% max overlap with each gt is loaded from roidb
max_labels = class*ones(size(image_roidb.class));
idx = find(max_overlaps > conf.fg_thresh);
image_id = image_roidb.image_id;
if conf.ignore_invalid% find invalid region::
    [iou_invalid, mask] = conf.invalid_read(image_id, rois);
    idx_rm = iou_invalid >= conf.bg_thresh_hi;
%     idx = setdiff(idx, find(idx_rm));
    if 0
        idx = find(ratio>0.5);
        idx = idx(1:2:60);
        figure(1)
        imshow(im)
        bia.plot.bb([],bia.convert.bb(rois(idx, :),'c2m'))
        figure(2)
        imshow(mask)
        bia.plot.bb([],bia.convert.bb(rois(idx, :),'c2m'))
    end
    %     idx_rm = intersect(bg_inds, idx_rm);
    %     bg_inds = setdiff(bg_inds, idx_rm);
end
rois = single(rois);% ensure ROIs are floats
bbox_targets = zeros(size(rois, 1), 5, 'single');
gt_inds = find(max_overlaps == 1);% Indices of ground-truth ROIs


if ~isempty(gt_inds)
    ex_inds = find(max_overlaps >= conf.bbox_thresh);% Indices of examples for which we try to make predictions
    ex_gt_overlaps = cpn.utils.boxoverlap(rois(ex_inds, :), rois(gt_inds, :));% Get IoU overlap between each ex ROI and gt ROI
    if ~isfield(conf, 'seg')
        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));
    end
    
    % Find which gt ROI each ex ROI has max overlap with:
    % this will be the ex ROI's gt target
    [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
    gt_rois = rois(gt_inds(gt_assignment), :);
    ex_rois = rois(ex_inds, :);
    
    [regression_label] = cpn.utils.fast_rcnn_bbox_transform(ex_rois, gt_rois);
    
    bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    if conf.ignore_invalid
        bbox_targets(idx_rm, 1) = -1;
    end
end

% Select foreground ROIs as those with >= fg_thresh overlap
is_fg = max_overlaps >= conf.fg_thresh;
% Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;

% check if there is any fg or bg sample. If no, filter out this image
is_valid = true;
if ~any(is_fg | is_bg)
    is_valid = false;
end
end


function [seg_masks, seg_weights, markers, stats] = compute_seg_targets(conf, image_id, rois)
%     rois: [ymin, xmin, ymax, xmax]
mask_sz = conf.mask_sz;
mask_type = conf.mask_type;% 1(classify all cell pixels), 2(classify only 1 cell pixels)
debug_disp = 0;

[~,~,mask,markers] = conf.gtread(image_id);

r   = round(rois);
sz  = size(mask);
N   = size(rois,1);
r(:,[1,2]) = min(repmat([sz(2) sz(1)]-1, size(r,1), 1), max(1, r(:,[1,2])) );%-1 is here to ensure that there is no empty ROI box
r(:,[3,4]) = min(repmat([sz(2) sz(1)]  , size(r,1), 1), r(:,[3,4]));

seg_masks = zeros(N, mask_sz^2);
seg_weights = ones(N, mask_sz^2);

% weight computation
use_weights = conf.use_weights;
sig         = 3;
if use_weights
    w = boundarymask(mask)>0;
    w = bwdist(w);
    w0 = exp(-(w.^2/(2*sig^2)));
end

for i=1:size(rois,1)
    ri = r(i,:);
    mask_roi = mask(ri(2):ri(4), ri(1):ri(3));
    if use_weights
        w0_roi = w0(ri(2):ri(4), ri(1):ri(3));
        w0_roi_rsz = imresize(w0_roi,[mask_sz, mask_sz],'nearest')';
        seg_weights(i,:) = w0_roi_rsz(:);
    end
    mask_roi_rsz = imresize(single(mask_roi), [mask_sz, mask_sz], 'nearest')';% transpose mask
    if mask_type == 1% detect all cells pixels
        seg_masks(i,:)      = mask_roi_rsz(:) > 0;
    elseif mask_type == 2% detect pixels of main cell only
        [mask_roi_rsz] = bia.seg.largest_obj(mask_roi_rsz);
        mask_back = imresize(mask_roi_rsz', [ri(4)-ri(2)+1, ri(3)-ri(1)+1], 'nearest');
        stats(i) = bia.stats.roi_stats(mask_back, ri([2 4 1 3]), sz);
        seg_masks(i,:) = mask_roi_rsz(:);
    end
    if debug_disp
        im = conf.imread(image_id);
        figure(1)
        imshow(bia.convert.l2rgb(mask))
        bia.plot.bb([],bia.convert.bb(ri,'c2m'))
        title('Image with ROI')
        figure(2)
        subplot(1,4,1);
        im_roi = bia.prep.norm( im(ri(2):ri(4), ri(1):ri(3), :));
        imshow(im_roi, [])
        subplot(1,4,2);
        imshow(reshape(seg_weights(i,:), mask_sz, mask_sz)', [])
        title('Weights for ROI')
        subplot(1,4,3)
        imshow(reshape(seg_masks(i,:), mask_sz, mask_sz)', [])
        title('Target Mask for ROI')
        subplot(1,4,4)
        imshow(bia.convert.l2rgb(mask_roi))
        title('GT Mask for ROI')
    end
end
if use_weights
    top_x = min(100,N);
    tmp = seg_masks(1:top_x, :)==1;
    fg  = sum(tmp(:));
    bg  = numel(tmp)-fg;
    w_bg= fg/bg;
    w_fg= 1;
    seg_weights(seg_masks==0) = seg_weights(seg_masks==0)+w_bg;
    seg_weights(seg_masks==1) = seg_weights(seg_masks==1)+w_fg;
end
end
