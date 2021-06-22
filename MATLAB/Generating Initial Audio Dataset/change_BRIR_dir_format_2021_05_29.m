clear all;
addpath(genpath('D:\T5-Backup\FYP\MATLAB\submodules'));
addpath(genpath('D:\T5-Backup\FYP\MATLAB\other_dependencies'));


base_dir = 'D:\FYP\MATLAB\BRIR\eBrIRD\Restaurant';

out_dir = 'D:\FYP\MATLAB\HRTF\TEST\eBrIRD\Restaurant';

subdir = {'IR1', 'IR2', 'IR3','IR4','IR5','IR6','IR7','IR8','IR9','IR10','IR11','IR12'};
file_type = '.wav';
for i = 1:length(subdir)
    wav_files = dir(strcat(base_dir,'\',subdir{i},'\','*',file_type));
    wav_files = {wav_files.name};
    for j = 1:length(wav_files)
        read_dir = strcat(base_dir,'\',subdir{i},'\',wav_files{j});
        [y, fs] = audioread(read_dir);
        write_file_name = strcat(subdir{i},'_',wav_files{j});
        write_out_dir = strcat(out_dir,'\',write_file_name);
        v_writewav(y,fs,write_out_dir,[],[],[],[]);
    end
end