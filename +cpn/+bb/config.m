function conf = config(varargin)
% fields needed for testing
% conf.{use_padding,output_map,feat_stride,anchors}

ip = inputParser;
ip = cpn.utils.config(ip);

% frcnn: unused parameters
ip.addParameter('redo_map',             true,   @islogical);%
ip.addParameter('anchors_center',       true,   @islogical);% 1: centers anchors, 0: faster r-cnn
ip.addParameter('anchors_kmeans',       true,   @islogical);% use kmeans for getting anchor
ip.addParameter('anchors_num',            9,    @isscalar);%
ip.addParameter('anchors_offset',        -1,    @isscalar);% where is the top left anchor centered. -1: compute it

ip.addParameter('anchor_ratios',          [0.5, 1, 2],    @ismatrix);% ratio list of anchors
ip.addParameter('anchor_scales',          [0.5 1],       @ismatrix);% scale list of anchors: actual scales are "cell_size*scales"

% training
ip.addParameter('stage', 'bb',    @isstr);% identifies the stage
ip.addParameter('balanced',    true, @islogical);% use equal # of pos/neg anchors: over-rides "fg_fraction"
ip.addParameter('bbox_transform', 'log',    @isstr);% dataset identifier
ip.addParameter('use_best_anchors',true,    @islogical);% some neutral anchors are used as positive if: IoU > bg_thresh_hi
ip.addParameter('rm_foi_bg',    true,   @islogical);% remove bg anchors outside field of interest
ip.addParameter('rm_foi_fg',    true,   @islogical);
ip.addParameter('foi_thresh',   0.5,   @isscalar);% only active when use_foi_bg | use_foi_fg are true

ip.addParameter('batch_size',     1024,     @isscalar);% Minibatch size ::256
ip.addParameter('fg_fraction',    0.5,      @isscalar);% Fraction of minibatch that is foreground labeled (class > 0)
ip.addParameter('bg_weight',       1.0,     @isscalar);% weight of background samples, when weight of foreground samples is 1.0
ip.addParameter('fg_thresh',       0.5,     @isscalar);% Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
ip.addParameter('bg_thresh_hi',    0.4,     @isscalar);% Overlap threshold for a ROI to be considered background (class = 0 if overlap in [bg_thresh_lo, bg_thresh_hi))% 0.4
ip.addParameter('bg_thresh_lo',    0.0,     @isscalar);% changed from 0.1 to 0.0 at 17:30 on 1.03.2017
ip.addParameter('use_padding',     false,   @islogical);% true -> conv layers of model use padding

% testing
ip.addParameter('drop_boxes_runoff_image',true, @islogical);% whether drop the anchors that has edges outside of the image boundary
ip.addParameter('test_stride', 0,    @isscalar);% 0: apply net just once, >0: pad image by this much and apply net, repeat this for full stride
% training + testing
% testing
ip.addParameter('train_seg_nms',   0.4,      @isscalar);
ip.addParameter('test_nms',        0.3,      @isscalar);
ip.addParameter('test_min_box_size',0,       @isscalar);
ip.addParameter('test_drop_boxes_runoff_image',   false,          @islogical);
ip.addParameter('feat_stride',     4,        @isscalar);% Stride in input image pixels at ROI pooling level (network specific) 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16

% solver
ip.addParameter('momentum', 0.9, @isscalar);
ip.addParameter('base_lr', 0.001, @isscalar);

ip.parse(varargin{:});
conf = ip.Results;


conf = cpn.io.set_paths(conf, 'bb');
conf.nms = struct('per_nms_topN',40000,'nms_overlap_thres',0.8,'after_nms_topN',2000,'nms_score',0.0,'nms_seg',0.7,'use_gpu',gpuDeviceCount>0);

% bb model
opts_mdl = struct('id',conf.mdl_id,'type','cpn','channels',conf.channels,'num_anchors',conf.anchors_num,'use_padding',conf.use_padding,'init','gaussian');
[net_model, train_net, test_net] = cpn.io.save_model(opts_mdl);

path_train_def = fullfile(conf.paths.dir, sprintf('train%s.prototxt', conf.paths.id));
path_solver = fullfile(conf.paths.dir, sprintf('solver%s.prototxt', conf.paths.id));

bia.caffe.save_net(train_net, path_train_def)
bia.caffe.save_net(test_net, fullfile(conf.paths.dir, sprintf('test%s.prototxt', conf.paths.id)))

solver = struct('net', path_train_def,'base_lr',conf.base_lr,'lr_policy','step','momentum',conf.momentum,'stepsize',conf.stepsize,...
    'max_iter',conf.max_iter,'weight_decay',conf.weight_decay,'gamma',0.1,...
    'display',1000,'snapshot',0);% solver setting
bia.caffe.save_solver(solver, path_solver)

conf.feat_stride = net_model.stride;
conf.anchors = cpn.bb.generate_anchors(conf);% anchors

end