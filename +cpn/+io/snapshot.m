function model_path = snapshot(conf, net, model_path, bbox_means, bbox_stds, bbox_name)
assert(~exist(model_path, 'file'))

if ~isempty(bbox_means) && ~isempty(bbox_stds)
    [weights_back, bias_back] = cpn.utils.bb_weights_norm(conf, net, bbox_name, bbox_means, bbox_stds);
end

net.save(model_path);
fprintf('Saved as %s\n', model_path);

if ~isempty(bbox_means) && ~isempty(bbox_stds)
    cpn.utils.bb_weights_norm(conf, net, bbox_name, bbox_means, bbox_stds, weights_back, bias_back);
end

end