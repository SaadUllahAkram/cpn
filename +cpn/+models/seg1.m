function [model, train_net, test_net] = seg1(opts)
model_id = opts.id;
mask_sz = opts.mask_sz;
channels = opts.channels;%3
numel_out = mask_sz*mask_sz;

opt_conv = struct('pad',1,'stride',1);
init_conv = struct('type','xavier');
init_std_rcnn = 0.001;% initial std for rcnn conv layer
input_size = [1 channels 600 600];

num_outputs = [32,64,128,256];
gs = [128 64 32];
num_fc = 256;
num_last = 16;
drop_ratio = 0.5;

import bia.caffe.layers.*
for TRAIN   = [0, 1]% network changes a bit  % TRAIN   = 1;% 1(Train), 0 (Test)
    %% Network definition
    net.layers = {};
    net.layers{end+1,1} = name(sprintf('cpn_seg_%d',model_id));
    net.layers{end+1,1} = input('data',input_size);
    net.layers{end+1,1} = input('rois',[1 5 1 1]);% [batch_ind, x1, y1, x2, y2] zero-based indexing
    if TRAIN
        net.layers{end+1,1} = input('seg_labels',[1 numel_out 1 1],'train');
        net.layers{end+1,1} = input('seg_weights',[1 numel_out 1 1],'train');
    end
    % 1st conv->relu->norm->pool
    net.layers(end+1:end+2) = conv('conv1','data',num_outputs(1),opt_conv,init_conv,1);
    net.layers{end+1} = norm('norm1','conv1');
    net.layers{end+1} = pool('pool1','norm1',3,2,'MAX',1);
    % 2nd conv->relu->norm->pool
    net.layers(end+1:end+2) = conv('conv2','pool1',num_outputs(2),opt_conv,init_conv,1);
    net.layers{end+1} = norm('norm2','conv2');
    net.layers{end+1} = pool('pool2','norm2',3,2,'MAX',1);
    % 3rd conv->relu->norm->pool
    net.layers(end+1:end+2) = conv('conv3','pool2',num_outputs(3),opt_conv,init_conv,1);
    net.layers{end+1} = norm('norm3','conv3');
    net.layers{end+1} = pool('pool3','norm3',3,2,'MAX',1);
    % 4th conv->relu->norm
    net.layers(end+1:end+2) = conv('conv4','pool3',num_outputs(4),opt_conv,init_conv,1);
    net.layers{end+1} = norm('norm4','conv4');
    % 1st deconv->relu->crop->sum
    net.layers{end+1,1} = deconv('up1','norm4',num_outputs(3),struct('group',gs(1),'stride',2,'kernel_size',4,'pad',1));
    net.layers{end+1,1} = relu('up1');
    
    net.layers{end+1,1} = crop('crop1',{'up1','norm3'});
    net.layers{end+1,1} = eltwise('sum1',{'norm3','crop1'});
    
    % 2nd deconv->relu->crop->sum
    net.layers{end+1,1} = deconv('up2','sum1',num_outputs(2),struct('group',gs(2),'stride',2,'kernel_size',4,'pad',1));
    net.layers{end+1,1} = relu('up2');
    
    net.layers{end+1,1} = crop('crop2',{'up2','norm2'});
    net.layers{end+1,1} = eltwise('sum2',{'norm2','crop2'});

    % 3rd deconv->relu->crop->sum
    net.layers{end+1,1} = deconv('up3','sum2',num_outputs(1),struct('group',gs(3),'stride',2,'kernel_size',4,'pad',1));
    net.layers{end+1,1} = relu('up3');
    
    net.layers{end+1,1} = crop('crop3',{'up3','norm1'});
    net.layers{end+1,1} = eltwise('sum3',{'norm1','crop3'});
    
    net.layers(end+1:end+2) = conv('conv_last','sum3',num_last,bia.utils.setfields(opt_conv,'kernel_size',1,'pad',0),init_conv,1);
    net.layers{end+1} = roi_pool('roi_pooled','conv_last','rois',mask_sz,1);
    net.layers{end+1} = fc('fc1','roi_pooled',num_fc(1),init_std_rcnn);
    net.layers{end+1} = relu('fc1');
    if drop_ratio > 0
        net.layers{end+1} = dropout('drop1', 'fc1', drop_ratio);
    end
    
    net.layers{end+1} = fc('seg_pred','fc1',numel_out*(2),init_std_rcnn);
    net.layers{end+1} = reshape('seg_score','seg_pred',[-1 2 numel_out 1]);
    
    if TRAIN
        net.layers{end+1} = reshape('rs_seg_labels','seg_labels',[0 1 -1 0]);
        net.layers{end+1} = accuracy('accuracy',{'seg_score','rs_seg_labels'});
        net.layers{end+1} = reshape('rs_seg_weights','seg_weights',[0 1 -1 0]);
        net.layers{end+1} = softmax_loss('loss_cls',{'seg_score','rs_seg_labels', 'rs_seg_weights'});
        train_net = net;
    else%[W H C B]
        net.layers{end+1} = softmax('seg_prob','seg_score');
        test_net = net;
    end
end
model = struct('id', model_id, 'input_size', input_size, 'seg_train', train_net, 'seg_test', test_net);
end