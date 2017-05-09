function [output_map, anchors_offset] = feat_map_size(conf, test_net_def_file, im_sz)

caffe_net = caffe.Net(test_net_def_file, 'test');
if ismember('data', caffe_net.blob_names)
    channels = size(caffe_net.blobs('data').get_data(), 3);
end

if nargin == 3
    im_sz = im_sz(:, [1 2]);
    input = [im_sz; im_sz(:,[2 1])];
else
    input = 100:conf.max_dim;%200 used as for phx there is 0x0 output for 100x100 input
    input = [input'; input'];
end
n = size(input,1);
output = nan(2*n,1);
for i = 1:size(input,1)
    net_inputs = {single(zeros([input(i,1:2), channels, 1]))};
    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    caffe_net.forward(net_inputs);

    cls_score = caffe_net.blobs('proposal_cls_score').get_data();
    output([i, i+n]) = [size(cls_score, 1), size(cls_score, 2)];
end
output_map = containers.Map([input(:,1); input(:,2)], output);

anchors_offset = get_stride(conf, caffe_net, 100);

conf.anchors_offset = anchors_offset;
bia.caffe.clear;
end


function anchors_offset = get_stride(conf, caffe_net, sz_im)
names = caffe_net.layer_names;
% set filter weights
uniform = 1;%1: set whole conv filter to '1', 0: set only the central element of conv filter to '1'
for i = 1:length(names)
    name = names{i};
    type = caffe_net.layers(name).type;
    if strcmp(type,'Convolution')
        w = caffe_net.params(name, 1).get_data();
        b = caffe_net.params(name, 2).get_data();
        if uniform
            wn = zeros(size(w));
            wn(:,:,1,1) = 1;
            bn = zeros(size(b));
        else
            wn = zeros(size(w));
            bn = zeros(size(b));
            sz = size(wn);
            mid = ceil(sz/2);
            wn(mid(1), mid(2)) = 1;
        end
        caffe_net.params(name,1).set_data(wn);
        caffe_net.params(name,2).set_data(bn);
    elseif strcmp(type,'Deconvolution')
    elseif strcmp(type,'InnerProduct')
    end
end


sz = sz_im;
receptive2 = zeros([sz, sz, conf.channels]);
receptive3 = zeros([sz, sz, conf.channels]);

for i=2*conf.feat_stride : conf.feat_stride : sz
    fprintf('%d ', i)
    dry = 1;
    j = 0;
    while (dry || cls_score(3, 3, 1) ~= 0 || cls_score(2, 2, 1) ~= 0)
        j = j+1;
        im_blob = single(zeros([sz, sz, conf.channels, 1]));
        im_blob(i,j) = 255;
        net_inputs{1} = im_blob;
        caffe_net.reshape_as_input(net_inputs);
        caffe_net.forward(net_inputs);
        cls_score = caffe_net.blobs('conv5').get_data();
        if cls_score(2, 2,1) > 0
            receptive2 = receptive2 + im_blob;
        end
        if cls_score(3, 3,1) > 0
            receptive3 = receptive3 + im_blob;
        end
        
        if dry && cls_score(2, 2,1) && cls_score(3, 3,1)
            dry = 0;
        end
        if conf.debug
            subplot(1,3,1);imshow(cls_score(:,:,1), [])
            subplot(1,3,2);imshow(receptive2, [])
            subplot(1,3,3);imshow(receptive3, [])
            drawnow
        end
    end
    if sum(receptive2(i,:)) > 0 && sum(receptive3(i,:)) > 0
        idx = find(receptive2(i,:));
        start2 = min(idx);
        end2 = max(idx);
        idx = find(receptive3(i,:));
        start3 = min(idx);
        end3 = max(idx);
        rc2 = start2+(end2-start2)/2;
        rc3 = start3+(end3-start3)/2;
        
        if conf.debug
            fprintf('Receptive field of (2,2): [%d - %d], center of recp. field: %1.1f\n', start2, end2, rc2)
            fprintf('Receptive field of (2,2): [%d - %d], center of recp. field: %1.1f\n', start3, end3, rc3)
        end
        anchors_offset(1) = [rc2-conf.feat_stride];
        anchors_offset(2) = [rc2];
        anchors_offset(3) = [rc3];
        
        assert(end3-end2 == conf.feat_stride)
        assert(start3-start2 == conf.feat_stride)
        assert(rc3-rc2 == conf.feat_stride)
        assert(anchors_offset(1) > 0)
        break
    end
end
end
