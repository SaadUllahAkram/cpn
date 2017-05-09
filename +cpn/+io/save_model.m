function [net_model, train_net, test_net] = save_model(opts_mdl)
model_id = opts_mdl.id;
if ismember(model_id, 1)
    [net_model, train_net, test_net] = cpn.models.bb1(opts_mdl);
elseif ismember(model_id, 2)
    [net_model, train_net, test_net] = cpn.models.seg1(opts_mdl);
end
end