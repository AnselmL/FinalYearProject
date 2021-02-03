clear all;
base_direc = 'E:\FYP\MATLAB\NOISEX-92\';

file_type = '.mat';
fs = 19980;
files = dir(strcat(base_direc,'*',file_type));

names = {files.name};


for i = 1:size(names,2)
    noise_name = erase(names{i},'.mat');
    file_direc = strcat(base_direc,names{i});
    audio = load(file_direc).(noise_name);
    
    v_writewav(audio,fs,strcat(base_direc,noise_name,'.wav'),[],[],[],[]);
end