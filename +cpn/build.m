function build()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

% Compile nms_mex
cpn_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(cpn_dir, '+utils', 'private');

if ~exist(fullfile(out_dir, 'nms_mex'), 'file')
  fprintf('Compiling nms_mex\n');
   cmd = sprintf('mex -O -outdir %s CXXFLAGS="\\$CXXFLAGS -std=c++11" -largeArrayDims %s -output nms_mex', out_dir, fullfile(out_dir, 'nms_mex.cpp'));
   eval(cmd)
end

if ~exist(fullfile(out_dir, 'nms_gpu_mex'), 'file')
   fprintf('Compiling nms_gpu_mex\n');
   addpath(out_dir);
   cpn.utils.nvmex(fullfile(out_dir, 'nms_gpu_mex.cu'), out_dir);
   delete('nms_gpu_mex.o');
end
