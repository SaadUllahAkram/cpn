function roidb = roidb_from_proposal(roidb, regions, varargin)

ip = inputParser;
ip.addParameter('keep_raw_proposal', true, @islogical);
ip.parse(varargin{:});
opts = ip.Results;

rois = roidb;
if ~opts.keep_raw_proposal
    % remove proposal boxes in roidb
    for i = 1:length(rois)  
        is_gt = rois(i).gt;
        rois(i).gt = rois(i).gt(is_gt, :);
        rois(i).overlap = rois(i).overlap(is_gt, :);
        rois(i).boxes = rois(i).boxes(is_gt, :);
        rois(i).class = rois(i).class(is_gt, :);
    end
end

% add new proposal boxes
for i = 1:length(rois)
    boxes = regions{i}(:, 1:4);
    boxes = bia.convert.bb(boxes, 'b2c');
    
    is_gt = rois(i).gt;
    gt_boxes = rois(i).boxes(is_gt, :);
    gt_classes = rois(i).class(is_gt, :);
    all_boxes = cat(1, rois(i).boxes, boxes);

    num_gt_boxes = size(gt_boxes, 1);
    num_boxes = size(boxes, 1);
    num_classes = size(rois(i).overlap, 2);
    
    rois(i).gt = cat(1, rois(i).gt, false(num_boxes, 1));
    rois(i).overlap = cat(1, rois(i).overlap, zeros(num_boxes, num_classes));
    rois(i).boxes = cat(1, rois(i).boxes, boxes);
    rois(i).class = cat(1, rois(i).class, ones(num_boxes, 1));% assign clas '1' to all proposals
    for j = 1:num_gt_boxes
        overlaps = cpn.utils.boxoverlap(all_boxes, gt_boxes(j, :));
        overlap_roidb = full(rois(i).overlap(:, gt_classes(j)));% overlaps from roidb
        rois(i).overlap(:, gt_classes(j)) = max(overlap_roidb, overlaps);
    end
%     figure;
%     imshow(cpn.io.imread(rois(i).image_id))
%     bia.plot.bb([],boxes(regions{i}(:, 5)>0.9,:),'g')
end
roidb = rois;

end