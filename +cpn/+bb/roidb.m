function [image_roidb, bbox_means, bbox_stds] = roidb(conf, roidbs, bbox_means, bbox_stds)

if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end

if ~iscell(roidbs)
    roidbs = {roidbs};
end
roidbs = roidbs(:);

n_datasets = length(roidbs);
image_roidb = cell(n_datasets,1);
for i=1:n_datasets% if multiple imdb/roidb are combined
    y = roidbs{1};
    image_roidb{i,1} = arrayfun(@(z) ... //@([1:length(x.image_ids)])
        struct('image_id', z.image_id, 'im_size', z.sz, ...
        'boxes', z.boxes(z.gt, :), 'class', z.class(z.gt, :), 'bbox_targets', []), ...
        y, 'UniformOutput', true);
end
image_roidb = cat(1, image_roidb{:});

% enhance roidb to contain bounding-box regression targets
[image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
end


function [roidb, means, stds] = append_bbox_regression_targets(conf, roidb, means, stds)
% means and stds -- (k+1) * 4, include background class
num_images = length(roidb);
% Infer number of classes from the number of columns in gt_overlaps
roidb_cell = num2cell(roidb, 2);
% bbox_targets = cell(num_images, 1);
disp_iter = max(1, round(num_images/20));
fprintf('%d::', num_images)
time_str = '';
time_start = tic;
for i = 1:num_images
    if rem(i, disp_iter) == 0
        fprintf(repmat('\b', 1, length(time_str)))
        time_str = sprintf('%3.0f%%: %4.1fsec', 100*i/num_images, toc(time_start));
        fprintf('%s', time_str)
    end
    [anchors, im_scales] = cpn.bb.get_anchors(conf, roidb_cell{i}.im_size);
    assert(im_scales == 1)
    
    gt_rois = roidb_cell{i}.boxes;
    gt_labels = roidb_cell{i}.class;
    roidb(i).bbox_targets{1} = compute_targets(conf, gt_rois, gt_labels, anchors, roidb_cell{i}, im_scales);
    % bbox_targets{i} = cellfun(@(x, y) compute_targets(conf, scale_rois(gt_rois, roidb_cell{i}.im_size, y), gt_labels,  x, roidb_cell{i}, y), ...
    %     anchors, im_scales, 'UniformOutput', false);
end
fprintf('\n')
clear roidb_cell;
% for i = 1:num_images
%     roidb(i).bbox_targets = bbox_targets{i};
% end
clear bbox_targets;

if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
    % Compute values needed for means and stds: var(x) = E(x^2) - E(x)^2
    class_counts = zeros(1, 1) + eps;
    sums = zeros(1, 4);
    squared_sums = zeros(1, 4);
    for i = 1:num_images
        targets = roidb(i).bbox_targets{1};
        gt_inds = find(targets(:, 1) > 0);
        if ~isempty(gt_inds)
            class_counts = class_counts + length(gt_inds);
            sums = sums + sum(targets(gt_inds, 2:end), 1);
            squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
        end
    end
    means = bsxfun(@rdivide, sums, class_counts);
    stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
end
% Normalize (remove mean and divide by std) targets
for i = 1:num_images
    targets = roidb(i).bbox_targets{1};
    gt_inds = find(targets(:, 1) > 0);
    if ~isempty(gt_inds)
        roidb(i).bbox_targets{1}(gt_inds, 2:end) = bsxfun(@minus, roidb(i).bbox_targets{1}(gt_inds, 2:end), means);
        roidb(i).bbox_targets{1}(gt_inds, 2:end) = bsxfun(@rdivide, roidb(i).bbox_targets{1}(gt_inds, 2:end), stds);
    end
end
end


function [bbox_targets, bbox_weights] = compute_targets(conf, gt_rois, gt_labels, ex_rois, image_roidb, im_scale)
% output: bbox_targets
%   positive: [class_label, regression_label]
%   ingore: [0, zero(regression_label)]
%   negative: [-1, zero(regression_label)]
if isempty(gt_rois)
    bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
    bbox_targets(:, 1) = -1;
    return;
end

% ensure gt_labels is in single
gt_labels = single(gt_labels);
assert(all(gt_labels > 0));

% calc overlap between ex_rois(anchors) and gt_rois
ex_gt_overlaps = cpn.utils.boxoverlap(ex_rois, gt_rois);

% drop anchors which run out off image boundaries, if necessary
if conf.drop_boxes_runoff_image
    contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
    ex_gt_overlaps(~contained_in_image, :) = 0;
end

% for each ex_rois(anchors), get its max overlap with all gt_rois
[ex_max_overlaps, ex_assignment] = max(ex_gt_overlaps, [], 2);

% for each gt_rois, get its max overlap with all ex_rois(anchors), the ex_rois(anchors) are recorded in gt_assignment
% gt_assignment will be assigned as positive (assign a rois for each gt at least)
[gt_max_overlaps, gt_assignment] = max(ex_gt_overlaps, [], 1);
if conf.use_best_anchors
    gt_max_overlaps(gt_max_overlaps <= max(0, conf.bg_thresh_hi)) = -1;% exclude best matched anchors with iou below threshold from pos space
    % ex_rois(anchors) with gt_max_overlaps maybe more than one, find them as (gt_best_matches)
    [gt_best_matches, gt_ind] = find(bsxfun(@eq, ex_gt_overlaps, [gt_max_overlaps]));
else
    gt_best_matches = [];
end
% Indices of examples for which we try to make predictions both (ex_max_overlaps >= conf.fg_thresh) and gt_best_matches are assigned as positive examples
fg_inds = unique([find(ex_max_overlaps >= conf.fg_thresh); gt_best_matches]);

% Indices of examples for which we try to used as negtive samples
% the logic for assigning labels to anchors can be satisfied by both the positive label and the negative label
% When this happens, the code gives the positive label precedence to pursue high recall
bg_inds = setdiff(find(ex_max_overlaps < conf.bg_thresh_hi & ex_max_overlaps >= conf.bg_thresh_lo), fg_inds);
if conf.drop_boxes_runoff_image
    contained_in_image_ind = find(contained_in_image);
    fg_inds = intersect(fg_inds, contained_in_image_ind);
    bg_inds = intersect(bg_inds, contained_in_image_ind);
end

% filter pos/neg anchors
rm_bg_foi = [];
rm_fg_foi = [];
if conf.rm_foi_bg || conf.rm_foi_fg
    [iou_invalid, invalid, gt_rois_verify] = conf.invalid_read(image_roidb.image_id, ex_rois);
    assert(isequal(size(invalid), image_roidb.im_size),'resizing issue')
    assert(isequal(gt_rois, gt_rois_verify),'gt rois computed on the fly and those in imdb differ')
    
    idx_rm = find(iou_invalid >= conf.foi_thresh);
    if conf.rm_foi_bg
        rm_bg_foi = intersect(bg_inds, idx_rm);
        bg_inds = setdiff(bg_inds, idx_rm);
    end
    if conf.rm_foi_fg
        rm_fg_foi = intersect(fg_inds, idx_rm);
        fg_inds = setdiff(fg_inds, idx_rm);
    end
end

% Find which gt ROI each ex ROI has max overlap with: this will be the ex ROI's gt target
target_rois = gt_rois(ex_assignment(fg_inds), :);
src_rois = ex_rois(fg_inds, :);

% we predict regression_label which is generated by a non-linear transformation from src_rois and target_rois
if strcmp(conf.bbox_transform, 'log')
    [regression_label] = cpn.utils.fast_rcnn_bbox_transform(src_rois, target_rois);
elseif strcmp(conf.bbox_transform, 'log2')
    [regression_label] = fast_rcnn_bbox_transform_log2(src_rois, target_rois);
end
bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
bbox_targets(fg_inds, :) = [gt_labels(ex_assignment(fg_inds)), regression_label];
bbox_targets(bg_inds, 1) = -1;
if conf.debug % debug
    [invalid, ~, mask] = conf.gtread(image_roidb.image_id);
    im = conf.imread(image_roidb.image_id);
    im = repmat(im(:,:,1), [1 1 3]);
    im(2*numel(invalid)+find(invalid == 1)) = 255;
    im = bia.draw.boundary(struct('alpha',0.7), im, mask);
    imshow(im);
    hold on;
    
    % data = struct('im', im, 'gt', gt_rois, 'pos', ex_rois(fg_inds, :), 'neg', ex_rois(bg_inds, :));save('cpn_fig.mat', 'data');% saves data for plotting fig
    num_disp = min([20, length(fg_inds)]);
    
    plot_bb(target_rois, num_disp, 'm','-')% target boxes ==  GT
    plot_bb(ex_rois(bg_inds,:), num_disp, 'r','-')% neg anchors
    plot_bb(ex_rois(fg_inds,:), num_disp, 'g','-')% pos anchors
    plot_bb(ex_rois(rm_bg_foi,:), num_disp, 'r','--')% removed (invalid/uncertain) BG anchors
    plot_bb(ex_rois(rm_fg_foi,:), num_disp, 'g','--')% removed (invalid/uncertain) FG anchors
    
    
    %figure
    %imshow(im);hold on;
    %[anchors, im_scales,shift_x,shift_y] = cpn.bb.get_anchors(conf, image_roidb.im_size);
    %plot_bb(ex_rois, size(ex_rois,1), 'g','-')
    %scatter(shift_x(:), shift_y(:),'filled')    
    % plot_bb(gt_rois, num_disp, 'm')% GT
    %hold off;
    drawnow
    pause(1)
end
bbox_targets = sparse(bbox_targets);
if nargout > 1
    bbox_weights = zeros(size(bbox_targets, 1), 1);
    bbox_weights(fg_inds) = 1-ex_max_overlaps(fg_inds);
end
end


function plot_bb(rois, num_disp, col, line_style)
n = size(rois,1);
if n > 0
    rois = rois(randperm(n, min(n, num_disp)), :);
    cellfun(@(x) rectangle('Position',bia.convert.bb(x,'c2b'),'EdgeColor',col,'LineStyle',line_style),num2cell(rois,2));
end
end


function contained = is_contain_in_image(boxes, im_size)
contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
contained = all(contained, 2);
end
