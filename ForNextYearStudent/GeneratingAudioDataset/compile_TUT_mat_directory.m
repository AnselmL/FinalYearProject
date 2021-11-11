clear all;
base_dir = 'E:\FYP\MATLAB\TUT-noise';

Folders = {'TUT-acoustic-scenes-2017-development.audio.1','TUT-acoustic-scenes-2017-development.audio.2','TUT-acoustic-scenes-2017-development.audio.3',...
           'TUT-acoustic-scenes-2017-development.audio.4','TUT-acoustic-scenes-2017-development.audio.5','TUT-acoustic-scenes-2017-development.audio.6',...
           'TUT-acoustic-scenes-2017-development.audio.7','TUT-acoustic-scenes-2017-development.audio.8','TUT-acoustic-scenes-2017-development.audio.9',...
           'TUT-acoustic-scenes-2017-development.audio.10'};
file_type = '.wav';
index = 1;
for i = 1:size(Folders,2)
    sub_dir = strcat(base_dir,'\',Folders{i});
    files = dir(sub_dir);
    files(ismember( {files.name}, {'.', '..'})) = [];
    dirFlags = [files.isdir];
    subFolders = files(dirFlags);
    subFolders = {subFolders.name};
    if isempty(subFolders)
        wav_files = {files.name};
        for l = 1:size(wav_files,2)
            wav_dir = strcat(sub_dir,'\',wav_files{l});
            TUT_wav_dirs{index,1} = wav_dir;
            index = index + 1;
        end
    else
        for j = 1:size(subFolders,2)
            subsub_dir = strcat(sub_dir,'\',subFolders{j});
            files = dir(subsub_dir);
            files(ismember( {files.name}, {'.', '..'})) = [];
            dirFlags = [files.isdir];
            subsubFolders = files(dirFlags);
            subsubFolders = {subsubFolders.name};
            if isempty(subsubFolders)
                wav_files = {files.name};
                for l = 1:size(wav_files,2)
                    wav_dir = strcat(subsub_dir,'\',wav_files{l});
                    TUT_wav_dirs{index,1} = wav_dir;
                    index = index + 1;
                end
            else
                for k = 1:size(subFolders,2)
                    subsub_dir = strcat(sub_dir,'\',subFolders{k});
                    files = dir(subsub_dir);
                    files(ismember( {files.name}, {'.', '..'})) = [];
                    dirFlags = [files.isdir];
                    subsubFolders = files(dirFlags);
                    subsubFolders = {subsubFolders.name};
                    if isempty(subsubFolders)
                        wav_files = {files.name};
                        for l = 1:size(wav_files,2)
                            wav_dir = strcat(subsub_dir,'\',wav_files{l});
                            TUT_wav_dirs{index,1} = wav_dir;
                            index = index + 1;
                        end
                    else
                        for m = 1:size(subsubFolders,2)
                            subsubsub_dir = strcat(subsub_dir,'\',subsubFolders{m});
                            wav_files = dir(strcat(subsubsub_dir,'\','*',file_type));
                            wav_files = {wav_files.name};
                            bob = 1;
                            for l = 1:size(wav_files,2)
                                wav_dir = strcat(subsubsub_dir,'\',wav_files{l});
                                TUT_wav_dirs{index,1} = wav_dir;
                                index = index + 1;
                            end
                        end
                    end

                end
            end
        end
    end
    
end
save('TUT_wav_dirs','TUT_wav_dirs');
%names = {subFolders.name};