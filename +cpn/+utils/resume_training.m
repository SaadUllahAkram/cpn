function [iter_max_path, iter_max, solver] = resume_training(train_dir, solver)
% 
% Output:
%     model_max : iter uptil which training has been done
% 

if strcmp(train_dir, '\') || strcmp(train_dir, '/')
    train_dir = train_dir(1:end-1);
end

[~, max_iters_orig] = update_solver(1, solver, 0);
[iter_max, iter_max_path] = find_last_model(train_dir);

% training has already finished
if iter_max >= max_iters_orig
    iter_max_path   = '';
    iter_max        = max_iters_orig;
    return
end

final_path = fullfile(train_dir, 'final');
if ~exist(final_path, 'file')
    final_path = fullfile(train_dir, sprintf('iter_%d', iter_max));
end

if iter_max > 0
    fprintf('##################\n##################\n RESUMING TRAINING FROM: iter: %d, model: %s \n##################\n##################\n',iter_max, iter_max_path)
    backup_old_model(final_path, iter_max, 0);
    solver = update_solver(0, solver, iter_max);
end

end

function [solver, max_iters_orig] = update_solver(flag, solver, iter_max)
% adjusts solver so that it can cope with 
%     flag: 1, just read the max iters and step size
%           else backup and modify solver file
% 

solver_orig = solver;
if flag == 1    
else
    solver = backup_old_model(solver_orig, iter_max, 1);
end
f     = fopen(solver_orig, 'r');
tline = fgets(f);
lines = cell(0);
lines{end+1} = tline;
while ischar(tline)
    tline = fgets(f);
    if ~isempty(strfind(tline, 'stepsize:')) 
        step    = str2double(tline(10:end));
        step    = step-iter_max;
        tline   = sprintf('stepsize: %d\n', step);
    elseif ~isempty(strfind(tline, 'max_iter:'))
        max_iters_orig    = str2double(tline(10:end));
        max_iters    = max_iters_orig-iter_max;
        tline   = sprintf('max_iter: %d\n', max_iters);        
        if flag ~= 1
            warning('Step size: %d, Max iterations: %d. Learning Rate may drop more than expected\n', step, max_iters)
        end
    end
    lines{end+1} = tline;
end
fclose(f);
if flag ~= 1
    f       = fopen(solver, 'w+');
    for i=1:length(lines)
        fprintf(f, '%s', lines{i});
    end
    fclose(f);
end
end

function [iter_max, iter_max_path] = find_last_model(train_dir)
% return the id and path of last saved model

list_models = dir([train_dir, filesep, 'iter_*']);
iter_max    = 0;
model_idx   = 1;
if length(list_models) > 2
    for i=1:length(list_models)
        cur_iter = str2double(list_models(i).name(6:end));
        if isnan(cur_iter)
            continue
        end
        if cur_iter > iter_max
            iter_max = cur_iter;
            model_idx = i;
        end
    end
    iter_max_path = fullfile(train_dir, list_models(model_idx).name);
else
    iter_max_path = '';
end

end


function final_path_new = backup_old_model(final_path, iter_max, dry)
% backs up 'final' model file to 'final_iter_iterNum_uniqueId' in case sth goes wrong
if nargin < 3
    dry = 1;
end
[train_dir, filename, ext] = fileparts(final_path);
func = @(x) fullfile(train_dir, sprintf('backup_%d_%03d_%s%s', x, iter_max, filename, ext));
if strcmp(filename(1:5), 'iter_')
    copy = 1;
else
    copy = 0;
end
% % if strcmp(filename(1:5), 'iter_')
% %    final_path_new = [final_path, '_backup'];
% %    copyfile(final_path, final_path_new)
% %    final_path = final_path_new;
% % end
if exist(final_path, 'file')
    i = 1;
    final_path_new = func(i);%[final_path, sprintf('_iter_%d_%d', iter_max, i)];
    while exist(final_path_new, 'file')
        i=i+1;
        final_path_new = func(i);
    end
    if ~dry
        if copy
            copyfile(final_path, final_path_new)
        else
            movefile(final_path, final_path_new)
        end
    end
end
end