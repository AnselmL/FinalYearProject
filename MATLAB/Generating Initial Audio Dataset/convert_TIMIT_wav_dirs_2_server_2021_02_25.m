clear all;

TIMIT_wav_dirs = load('TIMIT_wav_dirs').TIMIT_wav_dirs;
TIMIT_wav_dirs_server = cell(size(TIMIT_wav_dirs));
for i = 1:size(TIMIT_wav_dirs,1)
    temp = strrep(TIMIT_wav_dirs{i,1},'\','/');
    TIMIT_wav_dirs_server{i,1} = strrep(temp,'E:/FYP/MATLAB/','/home/al5517/');
end

save('TIMIT_wav_dirs_server','TIMIT_wav_dirs_server');
    