clear all;

measured_BRIR_wav_dirs = load('measured_BRIR_wav_dirs').BRIR_wav_dirs;
BRIR_wav_dirs_server = cell(size(measured_BRIR_wav_dirs));
for i = 1:size(measured_BRIR_wav_dirs,1)
    temp = strrep(measured_BRIR_wav_dirs{i,1},'\','/');
    BRIR_wav_dirs_server{i,1} = strrep(temp,'E:/FYP/MATLAB/','/home/al5517/');
end

save('measured_BRIR_wav_dirs_server','BRIR_wav_dirs_server');