function [im, mask, invalid] = cpn_crop_im(im, mask, invalid, flag)
% flag : 0 (no cropping), 76(select the middle 1110 rows), 1(select the top half+25 rows), 2(select bottom half+25 rows), 41:44(split image in 4 regions)

if nargin < 4
    flag = 0;
end
if nargin < 3
    invalid = [];
end
if nargin < 2
    mask = [];
end

sz = [max([size(im,1) size(mask,1) size(invalid,1)]) max([size(im,2) size(mask,2) size(invalid,2)])];
b = 25;
if isstruct(flag)% crop ids are in row major format
    split = flag;
    im_max_size = split.max_dim;
    y_splits = ceil(sz(1)/im_max_size);
    x_splits = ceil(sz(2)/im_max_size);
    splits   = y_splits*x_splits;
    if splits ~= split.num_splits
        error('num of splits is unexpected')
    end
    col_num = 1+mod(split.split_id, x_splits);%[2,3,4,5,...,1]: 1st part is at the end
    row_num = ceil(split.split_id/x_splits);
    
    rows = (max(1, (round(sz(1)/y_splits) * (row_num-1)) - b)):  (min(sz(1), (round(sz(1)/y_splits) * (row_num)) + b));
    cols = (max(1, (round(sz(2)/x_splits) * (col_num-1)) - b)):  (min(sz(2), (round(sz(2)/x_splits) * (col_num)) + b));
    if ~isempty(im)
        im = im(rows,cols,:);
    end
    if ~isempty(mask)
        mask = mask(rows,cols);
    end
    if ~isempty(invalid)
        invalid = set_invalid(invalid(rows,cols), b);
    end
else
if flag == 76% get the central part
    max_sz      = 1110;
    csz         = ceil(max(0, sz(1:2) - max_sz)/2);%how much to crop
    if sum(csz) == 0
        return;
    end
    if ~isempty(im)
        im          = im(csz(1)+1:end-csz(1), csz(2)+1:end-csz(2),:);
    end
    if ~isempty(mask)
        mask        = mask(csz(1)+1:end-csz(1), csz(2)+1:end-csz(2));
    end
    if ~isempty(invalid)
        invalid = set_invalid(invalid(csz(1)+1:end-csz(1), csz(2)+1:end-csz(2)), b);
    end
elseif ismember(flag, [1 2 41:44])
    if flag == 1% get the top part
        rows = 1:(round(sz(1)/2)+b);
        cols = 1:sz(2);
    elseif flag == 2% get the bottom part
        rows   = (1+round(sz(1)/2)-b) : sz(1);
        cols = 1:sz(2);
    elseif flag == 41
        rows = 1:(round(sz(1)/2)+b);
        cols = 1:(round(sz(2)/2)+b);
    elseif flag == 42
        rows = 1:(round(sz(1)/2)+b);
        cols = (round(sz(2)/2)-b):sz(2);
    elseif flag == 43
        rows = (round(sz(1)/2)-b):sz(1);
        cols = 1:(round(sz(2)/2)+b);
    elseif flag == 44
        rows = (round(sz(1)/2)-b):sz(1);
        cols = (round(sz(2)/2)-b):sz(2);
    end
    if ~isempty(im)
        im      = im(rows,cols,:);
    end
    if ~isempty(mask)
        mask    = mask(rows,cols,:);
    end
    if ~isempty(invalid)
        invalid = set_invalid(invalid(rows,cols), b);
    end
% % elseif flag == 1% get the top part
% %     ind_split   = round(sz(1)/2)+b;
% %     if ~isempty(im)
% %         im          = im(1:ind_split,:,:);
% %     end
% %     if ~isempty(mask)
% %         mask        = mask(1:ind_split,:);
% %     end
% %     if ~isempty(invalid)
% %         invalid = set_invalid(invalid(1:ind_split,:), b);
% %     end
% % elseif flag == 2% get the bottom part
% %     ind_split   = 1+round(sz(1)/2)-b;
% %     if ~isempty(im)
% %         im          = im(ind_split:end,:,:);
% %     end
% %     if ~isempty(mask)
% %         mask        = mask(ind_split:end,:);
% %     end
% %     if ~isempty(invalid)
% %         invalid = set_invalid(invalid(ind_split:end,:), b);
% %     end
end
end
end
function invalid = set_invalid(invalid, b)
invalid(1:b, :) = 1;
invalid(:, 1:b) = 1;
invalid(end-b+1:end, :) = 1;
invalid(:, end-b+1:end) = 1;
end