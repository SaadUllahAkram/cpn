function [rois, add_padd] = adjust_rois(conf, rois, sz)
% adjusts rois before segmentation
% 
% Inputs:
%     conf:
%     rois: corners [x1 y1 x2 y2]
%     im: 
% Outputs:
%     rois: []
%     add_padd:
%     im:
%     


add_padd = 0;
rois_local = rois;

padding_size = conf.roi_pad;
if padding_size == 0% padding is 0
elseif padding_size >= 1% padding is in pixels
    rois_local(:,1) = max(1, rois_local(:,1) - padding_size);
    rois_local(:,2) = max(1, rois_local(:,2) - padding_size);
    rois_local(:,3) = min(sz(2), rois_local(:,3) + padding_size);
    rois_local(:,4) = min(sz(1), rois_local(:,4) + padding_size);
elseif padding_size < 1% padding is in %
    w = rois_local(:,3)-rois_local(:,1);
    h = rois_local(:,4)-rois_local(:,2);

    assert(sum(w <= 0) == 0)
    assert(sum(h <= 0) == 0)

    pad_w   = padding_size*w;
    pad_h   = padding_size*h;

    rois_local(:,1) = max(1, rois_local(:,1) - pad_w);
    rois_local(:,2) = max(1, rois_local(:,2) - pad_h);
    rois_local(:,3) = min(sz(2), rois_local(:,3) + pad_w);
    rois_local(:,4) = min(sz(1), rois_local(:,4) + pad_h);
end
rois_local(rois_local<1) = 1;% rois are converted to 0-based indexing just b4 calling caffe
rois = rois_local;
end