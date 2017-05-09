function ip = config(ip)
% common parameters between CPN-BB and CPN-SEG

% experiment

% train + test
ip.addRequired('dataset',       @isstr);% dataset name
ip.addRequired('train_seq',     @ismatrix);% training seq: 0-> use all sequences
ip.addRequired('mdl_id',        @isscalar);% model identifier
ip.addRequired('exp_id',        @isscalar);% experiment identifier

ip.addParameter('conf_version',     1,  @isscalar);

% experiment level: do not influence training/testing results
ip.addParameter('do_val',       true, @islogical);
ip.addParameter('debug',        false, @islogical);
ip.addParameter('use_cache',      0,    @isscalar);%
ip.addParameter('val_interval',      1000,    @isscalar);% evaluate after the given interval
ip.addParameter('snapshot_interval', 5000, @isscalar);% save learnt model after the given iteration intervals
ip.addParameter('resume_train', false,      @islogical);% resumes training instead of starting from scratch, todo: resume even if the last run had completed
ip.addParameter('root_train',  '',  @isstr);% path where trained models and intermediary files will be saved

ip.addParameter('channels',    1,    @isscalar);%

% test only


% training only
% imdb settings
ip.addParameter('test_loss',   false,   @islogical);% 1: compute test loss, 0: compute val loss
ip.addParameter('val_ratio',    0,   @isscalar);% what proportion of images to use as validation set
ip.addParameter('gt_version', -1, @isscalar);% what ground truth to use: allow use of (semi-automatic) data augmentations to be used, -1: data specific
ip.addParameter('im_version', -1, @isscalar);% what images to use: allows use of different normalizations, -1: data specific
ip.addParameter('use_flips',    1, @isscalar);% use flips to augment data (include 90deg rotations as well)
ip.addParameter('rotations_interval',  0, @isscalar);% use rotations to augment data: 0-> no rotations, 10: rotations with interval of 10o
ip.addParameter('max_dim',    500,    @isscalar);% max size of an image during training. [larger images will be split]. Decrease it to reduce memory needed

ip.addParameter('imread',   @(x) cpn.io.imread(x),          @isstr);
ip.addParameter('gtread',   @(x) cpn.io.gtread(x),          @isstr);
ip.addParameter('invalid_read',   @(x,r) cpn.io.foiread(x,r),          @isstr);
ip.addParameter('image_means',     128,     @ismatrix);% mean image, in RGB order
ip.addParameter('scale',  -1,   @isscalar);% how to rescale the images: -1: data specific
ip.addParameter('foi',  25,    @isscalar);% field of non-interest (pixels around border without GT annotations)

% unfinished
ip.addParameter('deform',      0,    @isscalar);% how much deformation to do
ip.addParameter('ohem',        false, @islogical);% use OHEM for hard -ve mining

% solver
ip.addParameter('weight_decay', 0.0005, @isscalar);
ip.addParameter('stepsize', 30000, @isscalar);%30k
ip.addParameter('max_iter', 40000, @isscalar);%40k

end