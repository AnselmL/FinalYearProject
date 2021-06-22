clear all;
addpath(genpath('D:\FYP\MATLAB\API_MO-master'));
addpath(genpath('D:\FYP\MATLAB\submodules'));
addpath(genpath('D:\FYP\MATLAB\other_dependencies'));

SOFAstart;
%base_path = 'E:\FYP\MATLAB\eBrIRD';
base_path = 'C:\Users\ansel\Downloads';
%base_path = 'E:\FYP\MATLAB\MMHR-HRTF_sofa';
file_name = 'Kemar_HRTF_sofa.sofa';
%file_name = 'HRIR_CIRC360_NF150.sofa';
%file_name2 = 'hrtf_ci1.sofa';
full_path = strcat(base_path,'\',file_name);
%full_path2 = strcat(base_path,'\',file_name2);
hrtf = SOFAload(full_path);


%{

hrtf_data = hrtf.Data.IR;
src_positions = hrtf.SourcePosition;
fs = hrtf.Data.SamplingRate;
radius = src_positions(1,3);

base_directory = 'D:\FYP\MATLAB\HRTF\TEST\NF_KOELN';
%radius = 3.25;
az_angles = 0:5:355;
%el_angles = [-20,-10,0,10,20];
el_angles = 0;
%radius = 1;
%radius = [0.4,0.5,0.6,0.7,0.8,0.9,1];

index_array = zeros(length(az_angles),length(el_angles),length(radius));


for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            for l = 1:size(src_positions,1)
                if round(src_positions(l,1),1) == round(az_angles(i),1) && round(src_positions(l,2),1) == round(el_angles(j),1) && round(src_positions(l,3),-1) == round(radius(k),-1)
                    index_array(i,j,k) = l;
                end
            end
        end
    end
end

for i = 1:length(az_angles)
    for j = 1:length(el_angles)
        for k = 1:length(radius)
            found_hrtf = squeeze(hrtf_data(index_array(i,j,k),:,:)).';
            wav_string = strcat('az_',num2str(az_angles(i)),'_el_',num2str(el_angles(j)),'_radius_',num2str(radius(k)));
            out_wav_dir = strcat(base_directory,'\',wav_string,'.wav');
            v_writewav(found_hrtf,fs,out_wav_dir,[],[],[],[]);
        end
    end
end
%}
% list of azimuth angles you are interested in - i
% list of elevation angles you are interested in - j

% build array of indices (rows = i, columns = j) where the array element is
% (find first dimension for azimuth, second dimension for elevation)
% the index that the impulse response is found in

% then build string with azimuth, elevation and radius 

% Call directory based on what you want to call it