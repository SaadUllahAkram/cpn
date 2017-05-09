function [sequence, t, scale, gt_version, im_version, rot, flip, channels, split] = cpn_decode_name(fullname)
% returns info about the image augmentations/versions
% 
% Inputs:
%     fullname: file name
% Outputs:
%     sequence: 
%     t:
%     scale:
%     gt_version:
%     im_version:
%     rot: fun handle
%     flip: fun handle
%     channels:
%     split:
% 

% [~, fn, ~] = fileparts(fullname);
fn = fullname;
idx = strfind(fn, '-0');
idx = idx(1)+2;
sequence = fn(1:idx);

t = str2double(fn(idx+3:idx+5));

% rotation
idx = strfind(fn, '_rot');
if ~isempty(idx)
    idx = idx(1)+4;
    theta = str2double(fn(idx:idx+2));
    if theta == 90
        rot = @(x) rot90(x);
    else%if theta ~= 0% when rotating by rem(theta,90)~=0, set 0-padded pixels to mean/symmetric value
        rot = @(x) imrotate(x, theta, 'nearest');
    end
else
    rot = @(x) x;
end

% flips
if contains(fn, 'flipud')
    flip = @(x) flipud(x);
elseif contains(fn, 'fliplr')
    flip = @(x) fliplr(x);
elseif contains(fn, 'flipboth')
    flip = @(x) rot90(x,2);%180o rot
else
    flip = @(x) x;
end

% versions & scale & channels
idx = strfind(fn, '_gtv');
gt_version = str2double(fn(idx+4:idx+5));
idx = strfind(fn, '_imv');
im_version = str2double(fn(idx+4:idx+5));
idx = strfind(fn, '_scale');
scale = str2double(fn(idx+6:idx+9));
channels = 1 + 2*contains(fn, '_c3');

% splits
if contains(fn, '_imsplit_')
    idx = strfind(fn,'_imsplit_');
    vals = sscanf(fn(idx:end),'_imsplit_%d_%d_%d');
    split = struct('max_dim', vals(1), 'num_splits', vals(2), 'split_id', vals(3));
else
    split = struct('num_splits', 0, 'split_id', 0);
end

end