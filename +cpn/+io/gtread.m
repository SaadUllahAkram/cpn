function [invalid, bbox_tl_br, mask, markers] = gtread(im_path, opts_deform)

% % %     rois(i).gt              = true(num_gt, 1);
% % %     rois(i).overlap         = zeros(num_gt, max_num_classes);%num_gt x N_classes: overlap with the object
% % %     rois(i).overlap(:,class)= 1;
% % %     rois(i).class           = class*ones(num_gt, 1);%max_num_classes
% % %     rois(i).boxes           = bbox_tl_br; %num_gt x 4;% [x_pos, y_pos, x_pos2, y_pos2]
persistent gts
persistent invalids
persistent props

if nargin == 0
    gts = struct();
    invalids = struct();
    props = struct();
    return
end

if nargin == 2
    % bia.print.fprintf('red','Deforming GT\n')
    deform = opts_deform.map_deform;
    frames = opts_deform.frames;
    pad = opts_deform.pad;
    fields = fieldnames(frames);
    for i=1:length(fields)
        gt = gts.(fields{i});
        for t=frames.(fields{i}).t
            idx = find(gt.seg.info(:,1) == t);
            mask = bia.convert.stat2im(gt.seg.stats{idx}, gt.sz(t,:));
            invalid = invalids.(fields{i}){t};
            sz = size(mask);

            mask_n = padarray(mask, [pad pad],0);%'symmetric'
            invalid_n = padarray(invalid, [pad pad],1);

            mask_n = imwarp(mask_n, deform(1:sz(1)+2*pad,1:sz(2)+2*pad,:),'nearest','FillValues',0);
            invalid_n = imwarp(invalid_n, deform(1:sz(1)+2*pad,1:sz(2)+2*pad,:),'nearest','FillValues',1);

            mask_n = mask_n(pad+1:end-pad, pad+1:end-pad, :);
            invalid_n = invalid_n(pad+1:end-pad, pad+1:end-pad, :);
            invalid_n = set_invalid(invalid_n, props.(fields{i}).border);
            
            gt.seg.stats{idx} = regionprops(mask_n,'Area','BoundingBox','Centroid','PixelIdxList');
            gt.detect{t} = bia.convert.centroids(gt.seg.stats{idx});
            invalids.(fields{i}){t} = invalid_n;
            assert(sum(ismember(unique(invalid_n), [0 1])) == 2)
            assert(length(unique(invalid_n)) == 2)
        end
        gts.(fields{i}) = gt;
    end
    return
end
[dataset, t, scale, gt_version, im_version, rot, flip, ~, split] = cpn_decode_name(im_path);
d = strrep(dataset, '-', '_');
if ~isfield(gts, sprintf('%s', d))
    % fprintf('##################   Loading GT: %s   ##################\n', dataset)
    gts.(d) = bia.datasets.load(dataset,{'gt'}, struct('scale',scale,'version',[gt_version,im_version,0]));
    invalids.(d) = cell(gts.(d).T, 1);
    props.(d).border = gts.(d).foi_border;
    props.(d).rescale = scale;
end
gt     = gts.(d);
border = props.(d).border;

if isempty(invalids.(d){t})
    invalid = false(gt.sz(t,:));
    invalid = set_invalid(invalid, border);
    invalids.(d){t} = invalid;
else
    invalid = invalids.(d){t};
end

mask = bia.convert.stat2im(gt.seg.stats{gt.seg.info(:,1) == t}, gt.sz(t,:));
markers = round(gt.detect{t});

mask_c = zeros(gt.sz(t,:));
mask_c(sub2ind(gt.sz(t,:), markers(:,2),  markers(:,1))) = 1;

invalid = flip(invalid);
mask    = flip(mask);
mask_c  = flip(mask_c);


invalid = 1-invalid;
invalid = rot(invalid);
invalid = 1-invalid;
mask = rot(mask);
mask_c = rot(mask_c);

if isstruct(split) && split.num_splits > 1
    [~, mask, invalid] = cpn_crop_im([], mask, invalid, split);
    [~, mask_c] = cpn_crop_im([], mask_c, [], split);
end

stats_cent = regionprops(logical(mask_c), 'Centroid');
markers    = bia.convert.centroids(stats_cent);
bbox_tl_br = cpn_get_bbox(mask, invalid);
if isempty(markers)
    stats_cent = regionprops(logical(mask), 'centroid');
    markers = bia.convert.centroids(stats_cent);
end
% is(bia.convert.l2rgb(mask));
% bia.plot.bb([],stats)
end


function bbox_tl_br = cpn_get_bbox(mask, invalid)
stats       = regionprops(mask, 'BoundingBox', 'Area', 'PixelIdxList');
% remove cells which are completely inside invalid regions
stats       = cpn_remove_invalid_gt_obj(stats, invalid);
bbox_tl_br  = bia.convert.bb(stats, 's2c');% convert to VOC format (faster RCNN) [top left, bottom right] from [top left, width]
end


function stats = cpn_remove_invalid_gt_obj(stats, invalid)
N = length(stats);
rm_idx = false(N,1);
for i=1:N
    if sum(invalid(stats(i).PixelIdxList)) == length(stats(i).PixelIdxList)
        rm_idx(i) = 1;
    end
end
stats(rm_idx) = [];
end


function im = set_invalid(im, border)
im(1:border, :) = 1;
im(:, 1:border) = 1;
im(end-border+1:end, :) = 1;
im(:, end-border+1:end) = 1;
end