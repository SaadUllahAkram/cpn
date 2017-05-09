function anchors = generate_anchors(conf)
% returns anchors: 1) using k-means OR based on given box size/aspect ratio
% 

anchors_file = fullfile(conf.paths.dir, 'anchors.mat');
if ~exist(anchors_file, 'file')
    if conf.anchors_kmeans
        bia.print.fprintf('*red', 'creating anchors (kmeans)\n')
        bb = get_bbox(conf);
        anchors_pos = 1;
        bb = [bb; bb(:,2), bb(:,1)];
        [~, bb_kmeans] = kmeans(bb, conf.anchors_num, 'distance', 'cityblock');
        anchors = [anchors_pos-bb_kmeans(:,1)/2  anchors_pos-bb_kmeans(:,2)/2  bb_kmeans(:,1)/2+anchors_pos  bb_kmeans(:,2)/2+anchors_pos];
    else
        bb = get_bbox(conf);
        cell_size = round(mean(bb(:)));
        conf.anchor_scales = cell_size/round(conf.feat_stride);
        anchors = manual_anchors(conf);
    end
    save(anchors_file, 'anchors')
else
    load(anchors_file, 'anchors');
end

conf.anchors = anchors;
assert(conf.anchors_num == size(conf.anchors, 1), 'Number of anchors in settings is not consistent')

% imshow(zeros(200,200))
% hold on
% col = {'r','g','b','c','m','w'};
% for i=1:size(anchors,1)
%     rectangle('position',bia.convert.bb(anchors(i,:)+100,'c2m'),'edgecolor',col{rem(i,6)+1})
% end
end


function bb = get_bbox(conf)
gt = bia.datasets.load(sprintf('%s-%02d',conf.dataset,conf.train_seq), '',struct('scale',conf.scale));
stats = bia.datasets.stats(struct('verbose',0), gt);
bb = stats.bb_stats.bb;
end


function anchors = manual_anchors(conf)
if conf.anchors_center
    sz = (anchor_base_size-1)/2;
    base_anchor = [-sz, -sz, sz, sz];
else
    base_anchor = [1, 1, anchor_base_size, anchor_base_size];
end
ratio_anchors = ratio_jitter(base_anchor, conf.anchor_ratios);
anchors = cellfun(@(x) scale_jitter(x, conf.anchor_scales), num2cell(ratio_anchors, 2), 'UniformOutput', false);
anchors = cat(1, anchors{:});

save(anchor_cache_file, 'anchors');
end


function anchors = ratio_jitter(anchor, ratios)
ratios = ratios(:);

w = anchor(3) - anchor(1) + 1;
h = anchor(4) - anchor(2) + 1;
x_ctr = anchor(1) + (w - 1) / 2;
y_ctr = anchor(2) + (h - 1) / 2;
size = w * h;

size_ratios = size ./ ratios;
ws = round(sqrt(size_ratios));
hs = round(ws .* ratios);

anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end


function anchors = scale_jitter(anchor, scales)
scales = scales(:);

w = anchor(3) - anchor(1) + 1;
h = anchor(4) - anchor(2) + 1;
x_ctr = anchor(1) + (w - 1) / 2;
y_ctr = anchor(2) + (h - 1) / 2;

ws = w * scales;
hs = h * scales;

anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end