clear all;
base_dir = 'E:\FYP\MATLAB\TIMIT';
files = dir('E:\FYP\MATLAB\TIMIT');
%files(ismember( {files.name}, {'.', '..'})) = [];
%dirFlags = [files.isdir];
%subFolders = files(dirFlags);

Folders = {'TEST','TRAIN'};
file_type = '.wav';
index = 1;
for i = 1:size(Folders,2)
    sub_dir = strcat(base_dir,'\',Folders{i});
    files = dir(sub_dir);
    files(ismember( {files.name}, {'.', '..'})) = [];
    dirFlags = [files.isdir];
    subFolders = files(dirFlags);
    subFolders = {subFolders.name};
    for j = 1:size(subFolders,2)
        subsub_dir = strcat(sub_dir,'\',subFolders{j});
        files = dir(subsub_dir);
        files(ismember( {files.name}, {'.', '..'})) = [];
        dirFlags = [files.isdir];
        subsubFolders = files(dirFlags);
        subsubFolders = {subsubFolders.name};
        for k = 1:size(subsubFolders,2)
            subsubsub_dir = strcat(subsub_dir,'\',subsubFolders{k});
            wav_files = dir(strcat(subsubsub_dir,'\','*',file_type));
            wav_files = {wav_files.name};
            bob = 1;
            for l = 1:size(wav_files,2)
                wav_dir = strcat(subsubsub_dir,'\',wav_files{l});
                TIMIT_wav_dirs{index,1} = wav_dir;
                index = index + 1;
            end
                
        end
    end
                
    
end
%save('TIMIT_wav_dirs','TIMIT_wav_dirs');
%names = {subFolders.name};


count_train = 1;
count_test = 1;
for i = 1:size(TIMIT_wav_dirs,1)
    if i > 1680
        TIMIT_wav_dirs_train{count_train,1} = TIMIT_wav_dirs{i};
        count_train = count_train + 1;
    elseif i < 1681
        TIMIT_wav_dirs_test{count_test,1} = TIMIT_wav_dirs{i};
        count_test = count_test + 1;
    end
end

save('TIMIT_wav_dirs_split','TIMIT_wav_dirs_train','TIMIT_wav_dirs_test');