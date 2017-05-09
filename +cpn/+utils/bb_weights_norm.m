function [weights_back, bias_back] = bb_weights_norm(conf, net, bbox_layer, bbox_means, bbox_stds, weights_back, bias_back)
if nargin == 5
    if isfield(conf, 'anchors')
        anchor_size = size(conf.anchors, 1);
        bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
        bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);
    else
        bbox_stds_flatten = reshape(bbox_stds', [], 1);
        bbox_means_flatten = reshape(bbox_means', [], 1);
    end

    weights = net.layers(bbox_layer).params(1).get_data();
    bias = net.layers(bbox_layer).params(2).get_data();
    weights_back = weights;
    bias_back = bias;
    
    weights = bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds;
    bias = bias .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

    net.layers(bbox_layer).params(1).set_data(weights);
    net.layers(bbox_layer).params(2).set_data(bias);
else% restore net to original state
    net.layers(bbox_layer).params(1).set_data(weights_back);
    net.layers(bbox_layer).params(2).set_data(bias_back);
end
end