function [anchors, im_scales, shift_x, shift_y] = get_anchors(conf, im_size, ~, feature_map_size)

% only for fcn
if ~exist('feature_map_size', 'var')
    feature_map_size = [];
end
[anchors, im_scales, shift_x, shift_y] = proposal_locate_anchors_single_scale(im_size, conf, feature_map_size);
end


function [anchors, im_scale, shift_x, shift_y] = proposal_locate_anchors_single_scale(im_size, conf, feature_map_size)
im_scale = 1;
padding = conf.use_padding;

if isempty(feature_map_size)
    img_size = round(im_size * im_scale);
    output_size = cell2mat([conf.output_map.values({img_size(1)}), conf.output_map.values({img_size(2)})]);
else
    output_size = feature_map_size;
end
if isfield(conf,'anchors_offset') && conf.anchors_center
    shift_x = [0:(output_size(2)-1)] * conf.feat_stride + conf.anchors_offset(1);
    shift_y = [0:(output_size(1)-1)] * conf.feat_stride + conf.anchors_offset(1);
else
    shift_x = [0:(output_size(2)-1)] * conf.feat_stride;
    shift_y = [0:(output_size(1)-1)] * conf.feat_stride;
    if ~padding
        offset = 0.5*abs(im_size(1:2) - conf.feat_stride*(output_size(1:2)-1));
        shift_x = shift_x + offset(2)-0.5*conf.feat_stride;
        shift_y = shift_y + offset(1)-0.5*conf.feat_stride;
        assert(length(shift_x) == output_size(2))
        assert(length(shift_y) == output_size(1))
    end
end
[shift_x, shift_y] = meshgrid(shift_x, shift_y);

% concat anchors as [channel, height, width], where channel is the fastest dimension.
anchors = reshape(bsxfun(@plus, permute(conf.anchors, [1, 3, 2]), ...
    permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);

%   equals to
%     anchors = arrayfun(@(x, y) single(bsxfun(@plus, conf.anchors, [x, y, x, y])), shift_x, shift_y, 'UniformOutput', false);
%     anchors = reshape(anchors, [], 1);
%     anchors = cat(1, anchors{:});

end
