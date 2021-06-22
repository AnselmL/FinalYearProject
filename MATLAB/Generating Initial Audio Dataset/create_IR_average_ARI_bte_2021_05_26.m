clear all;
addpath(genpath('E:\FYP\MATLAB\API_MO-master'));
addpath(genpath('E:\FYP\MATLAB\submodules'));
addpath(genpath('E:\FYP\MATLAB\other_dependencies'));


%base_path = 'E:\FYP\MATLAB\eBrIRD';
base_path = 'E:\FYP\MATLAB\ARI_bte_sofa';
%file_name = 'hrtf b_ci1.sofa';
%file_name2 = 'hrtf_ci1.sofa';


file_type = '.sofa';
sofa_files = dir(strcat(base_path,'\','*',file_type));
sofa_files = {sofa_files.name};

IR_all = zeros(1550,2,256,length(sofa_files));
src_positions = zeros(1550,3,length(sofa_files));
SOFAstart;
for i = 1:length(sofa_files)
    full_path = strcat(base_path,'\',sofa_files{i});
    hrtf = SOFAload(full_path);
    IR_all(:,:,:,i) = hrtf.Data.IR;
    src_positions(:,:,i) = hrtf.SourcePosition;
    %rec_pos(:,:,i) = hrtf.ReceiverPosition;
    
    
    
end
fs = hrtf.Data.SamplingRate;
equal_check = isequal(src_positions(:,:,1),src_positions(:,:,2),src_positions(:,:,3),src_positions(:,:,4),src_positions(:,:,5),src_positions(:,:,6),src_positions(:,:,7),src_positions(:,:,8),src_positions(:,:,9));

final_IR = mean(IR_all,4);
hrtf_data = final_IR;
src_positions = mean(src_positions,3);

base_directory = 'E:\FYP\MATLAB\HRTF\ARI_bte';
az_angles = 0:2.5:357.5;
el_angles = -20:5:20;
radius = 1.2;

index_array = zeros(length(az_angles),length(el_angles),length(radius));


for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            for l = 1:size(src_positions,1)
                if src_positions(l,1) == az_angles(i) && src_positions(l,2) == el_angles(j) && src_positions(l,3) == radius(k)
                    index_array(i,j,k) = l;
                end
            end
        end
    end
end

for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            if mod(i,2) == 1 && index_array(i,j,k) ~= 0
                found_hrtf = hrtf_data(index_array(i,j,k),:,:);
                wav_string = strcat('az_',num2str(az_angles(i)),'_el_',num2str(el_angles(j)),'_radius_',num2str(radius(k)));
                out_wav_dir = strcat(base_directory,'\',wav_string,'.wav');
                found_hrtf = squeeze(found_hrtf).';
                v_writewav(found_hrtf,fs,out_wav_dir,[],[],[],[]);
            end
            if mod(i,2) == 1 && index_array(i,j,k) == 0
                if index_array(i-1,j,k) ~= 0 && index_array(i+1,j,k) ~= 0
                    found_hrtf = (hrtf_data(index_array(i-1,j,k),:,:) + hrtf_data(index_array(i+1,j,k),:,:))/2;
                    hrtf_data = cat(1,hrtf_data,found_hrtf);
                    src_positions = cat(1,src_positions,[az_angles(i),el_angles(j),radius(k)]);
                    wav_string = strcat('az_',num2str(az_angles(i)),'_el_',num2str(el_angles(j)),'_radius_',num2str(radius(k)));
                    out_wav_dir = strcat(base_directory,'\',wav_string,'.wav');
                    found_hrtf = squeeze(found_hrtf).';
                    v_writewav(found_hrtf,fs,out_wav_dir,[],[],[],[]);
                end
            end
        end
    end
end
index_array = zeros(length(az_angles),length(el_angles),length(radius));
for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            for l = 1:size(src_positions,1)
                if src_positions(l,1) == az_angles(i) && src_positions(l,2) == el_angles(j) && src_positions(l,3) == radius(k)
                    index_array(i,j,k) = l;
                end
            end
        end
    end
end

for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            if mod(i,2) == 1 && index_array(i,j,k) == 0
                    if index_array(i-2,j,k) ~= 0 && index_array(i+2,j,k) ~= 0
                        found_hrtf = (hrtf_data(index_array(i-2,j,k),:,:) + hrtf_data(index_array(i+2,j,k),:,:))/2;
                        hrtf_data = cat(1,hrtf_data,found_hrtf);
                        src_positions = cat(1,src_positions,[az_angles(i),el_angles(j),radius(k)]);
                        wav_string = strcat('az_',num2str(az_angles(i)),'_el_',num2str(el_angles(j)),'_radius_',num2str(radius(k)));
                        out_wav_dir = strcat(base_directory,'\',wav_string,'.wav');
                        found_hrtf = squeeze(found_hrtf).';
                        v_writewav(found_hrtf,fs,out_wav_dir,[],[],[],[]);
                    end
            end
        end
    end
end
az_angles = 0:5:355;
index_array = zeros(length(az_angles),length(el_angles),length(radius));
for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            for l = 1:size(src_positions,1)
                if src_positions(l,1) == az_angles(i) && src_positions(l,2) == el_angles(j) && src_positions(l,3) == radius(k)
                    index_array(i,j,k) = l;
                end
            end
        end
    end
end



%}