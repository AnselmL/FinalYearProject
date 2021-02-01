clear all;
TIMIT_wav_dirs = load('TIMIT_wav_dirs').TIMIT_wav_dirs;

count_train = 1;
count_val = 1;
count_test = 1;
for i = 1:size(TIMIT_wav_dirs,1)
    if i > 1680
        TIMIT_wav_dirs_train{count_train} = TIMIT_wav_dirs{i};
        count_train = count_train + 1;
    elseif (0 < i) && (i < 631)
        TIMIT_wav_dirs_val{count_val} = TIMIT_wav_dirs{i};
        count_val = count_val + 1;
    elseif (630 < i) && (i < 1681)   
        TIMIT_wav_dirs_test{count_test} = TIMIT_wav_dirs{i};
        count_test = count_test + 1;
    end
end

save('TIMIT_wav_dirs_split','TIMIT_wav_dirs_train','TIMIT_wav_dirs_val','TIMIT_wav_dirs_test');

