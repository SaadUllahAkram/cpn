function [model, train_net, test_net] = bb1(opts)

model_id = opts.id;
num_anchors = opts.num_anchors;%6
channels = opts.channels;%3
init = opts.init;

init_std = 0.01;% init std for conv layers
loss_w_class = 1;
loss_w_bbox = 10;

if strcmp(init, 'msra')
    init_para = struct('type','msra');
elseif strcmp(init, 'xavier')
    init_para = struct('type','xavier');
elseif strcmp(init, 'gaussian')
    init_para = struct('type','gaussian','std',init_std);
end

kern_pool = 3;
if model_id == 1
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512];
    WH_OUT = 142;% output size
elseif model_id == 101
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512]/4;
    WH_OUT = 142;% output size
elseif model_id == 102
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512]/2;
    WH_OUT = 142;% output size
elseif model_id == 103
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512]*2;
    WH_OUT = 142;% output size
elseif model_id == 104
    kern = [3 3 3 3 3 1 1];
    feats = [32 64 128 256 512]/4;
    WH_OUT = 144;% output size
elseif model_id == 105
    kern = [3 3 3 3 3 1 1];
    feats = [32 64 128 256 512]/2;
    WH_OUT = 144;% output size
elseif model_id == 106
    kern = [3 3 3 3 3 1 1];
    feats = [32 64 128 256 512]*2;
    WH_OUT = 144;% output size
elseif model_id == 107
    kern = [3 3 3 3 3 1 1];
    feats = [32 64 128 256 512];
    WH_OUT = 144;% output size
elseif model_id == 108
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512];
    WH_OUT = 142;% output size
elseif model_id == 109
    kern = [7 5 3 3 3 1 1];
    feats = [32 64 128 256 512];
    WH_OUT = 142;% output size
elseif model_id == 4
    kern = [3 3 3 3 3 1 1];
    feats = [32 64 128 256 512];
    WH_OUT = 144;% output size
end
padds = zeros(size(kern));
names_conv = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'};

WH = 600;% input size
input_size = [1 channels WH WH];

for TRAIN = [0, 1]% 1(Train net), 0 (Test net)
    names_data = {'data', 'labels', 'labels_weights', 'bbox_targets', 'bbox_loss_weights'};
    names_pool = {'pool1', 'pool2', 'pool3'};
    
    conv_props_names    = {'proposal_cls_score', 'proposal_bbox_pred'};
    reshape_props_names = {'proposal_cls_score_reshape', 'labels_reshape', 'labels_weights_reshape'};
    out_props_names     = {'loss', 'accuracy', 'loss_bbox'};
    out_test_props_names= {'proposal_cls_prob'};
    loss_name           = 'loss_cls';
    
    
    net.layers = {};
    net.layers{end+1,1} = bia.caffe.layers.name(sprintf('cpn_bb_%d',model_id));
    net.layers{end+1,1} = bia.caffe.layers.input('data',input_size);
    if TRAIN
        net.layers{end+1,1} = bia.caffe.layers.input(names_data{2}, [1 num_anchors   WH_OUT  WH_OUT]);
        net.layers{end+1,1} = bia.caffe.layers.input(names_data{3}, [1 num_anchors   WH_OUT  WH_OUT]);
        net.layers{end+1,1} = bia.caffe.layers.input(names_data{4}, [1 4*num_anchors   WH_OUT  WH_OUT]);
        net.layers{end+1,1} = bia.caffe.layers.input(names_data{5}, [1 4*num_anchors   WH_OUT  WH_OUT]);
    end
    % 1st conv->relu->norm->pool
    net.layers(end+1:end+2) = bia.caffe.layers.conv(names_conv{1},names_data{1},feats(1),struct('kernel_size',kern(1),'pad',padds(1)),init_para,1);
    net.layers{end+1} = bia.caffe.layers.norm(['norm',names_conv{1}],names_conv{1});
    net.layers{end+1} = bia.caffe.layers.pool(names_pool{1},['norm',names_conv{1}], kern_pool, 2, 'MAX',1);
    
    % 2nd conv->relu->norm->pool
    net.layers(end+1:end+2) = bia.caffe.layers.conv(names_conv{2},names_pool{1},feats(2),struct('kernel_size',kern(2),'pad',padds(2)),init_para,1);
    net.layers{end+1} = bia.caffe.layers.norm(['norm',names_conv{2}],names_conv{2});
    net.layers{end+1} = bia.caffe.layers.pool(names_pool{2},['norm',names_conv{2}], kern_pool, 2, 'MAX',1);
    
    % 3rd conv->relu->norm
    net.layers(end+1:end+2) = bia.caffe.layers.conv(names_conv{3},names_pool{2},feats(3),struct('kernel_size',kern(3),'pad',padds(3)),init_para,1);
    net.layers{end+1} = bia.caffe.layers.norm(['norm',names_conv{3}],names_conv{3});
    
    % 4th conv->relu->norm
    net.layers(end+1:end+2) = bia.caffe.layers.conv(names_conv{4},['norm',names_conv{3}],feats(4),struct('kernel_size',kern(4),'pad',padds(4)),init_para,1);
    net.layers{end+1} = bia.caffe.layers.norm(['norm',names_conv{4}],names_conv{4});

    % 5th conv->relu
    net.layers(end+1:end+2) = bia.caffe.layers.conv(names_conv{5},['norm',names_conv{4}],feats(5),struct('kernel_size',kern(5),'pad',padds(5)),init_para,1);
    
    % FC-Layers
    net.layers{end+1} = bia.caffe.layers.conv(conv_props_names{1},names_conv{5},2*num_anchors,struct('kernel_size',kern(6),'pad',padds(6)),init_para);
    net.layers{end+1} = bia.caffe.layers.conv(conv_props_names{2},names_conv{5},4*num_anchors,struct('kernel_size',kern(7),'pad',padds(7)),init_para);
    
    net.layers{end+1} = bia.caffe.layers.reshape(reshape_props_names{1}, conv_props_names{1}, [0 2 -1 0]);
    if TRAIN% train only layers
        net.layers{end+1} = bia.caffe.layers.reshape(reshape_props_names{2}, names_data{2}, [0 1 -1 0]);
        net.layers{end+1} = bia.caffe.layers.reshape(reshape_props_names{3}, names_data{3}, [0 1 -1 0]);
        % loss
        net.layers{end+1} = bia.caffe.layers.softmax_loss(out_props_names{1}, {reshape_props_names{1}, reshape_props_names{2}, reshape_props_names{3}}, loss_name, loss_w_class);
        net.layers{end+1} = bia.caffe.layers.accuracy(out_props_names{2}, {reshape_props_names{1}, reshape_props_names{2}});
        net.layers{end+1} = bia.caffe.layers.smooth_l1loss(out_props_names{3}, {conv_props_names{2}, names_data{4}, names_data{5}}, loss_w_bbox);
        train_net = net;
    else% test only layers
        net.layers{end+1}   = bia.caffe.layers.softmax(out_test_props_names{1}, reshape_props_names{1});
        test_net = net;
    end
end
model = struct('id',model_id,'stride',4,'num_anchors',num_anchors,'input_size',input_size,'output_size', [WH_OUT WH_OUT], 'cpn_train', train_net, 'cpn_test', test_net);

in = bia.caffe.check_size(model);
assert(in == WH_OUT, 'Size in middle layer was different')
assert( WH/WH_OUT == WH/WH_OUT );
end