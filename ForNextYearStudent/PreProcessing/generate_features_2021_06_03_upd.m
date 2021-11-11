%clear all;


addpath(genpath('E:\FYP\MATLAB\MBSTOI'));
addpath(genpath('E:\FYP\MATLAB\submodules'));
addpath(genpath('E:\FYP\MATLAB\other_dependencies'));

%specify audio input directory
audio_input_base_dir = 'E:\FYP\feature_preprocessing_input\directional_2021_11_11';

mixed_audio_input_name = 'mixed.wav';
clean_audio_input_name = 'clean.wav';
%Also specify features output directory
audio_output_base_dir = 'E:\FYP\feature_preprocessing_output\directional_2021_11_11';


%specifications for fft
fs_preprocessing        = 10000;    % Sampling rate
N_frame                 = 256;    	% Window support [samples].
K                       = 256;     	% FFT size [samples].
K_phase                 = 256;      % FFT size for phase features
dyn_range               = 40;      	% Speech dynamic range [dB].
intermediate_frame_size = 4224;



%calculating MFCC filter-band edges
fmin = 150;
fmax = 4500;
mel_filter_num = 15;

fmin_mel = hz2mel(fmin);
fmax_mel = hz2mel(fmax);

band_edges_mels = linspace(fmin_mel, fmax_mel, mel_filter_num+1);

band_edges_freqs = mel2hz(band_edges_mels);


%calculating one-third octave-band edges and filter
J = 15;
[H,~,fids] = thirdoct(fs_preprocessing,K,J,150);
H_trunc = H(:,4:end-20);


data_split = ["TRAIN", "TEST"];
for i = 1:length(data_split)
    current_subdir = strcat(audio_input_base_dir,'\',data_split(i));
    dir_list=dir(current_subdir);
    dirflags = [dir_list.isdir];
    subdir = dir_list(dirflags);
    subdir(ismember( {subdir.name}, {'.', '..'})) = [];  %remove . and ..
    subdirs = {subdir.name};
    subdirs_num = zeros(length(subdirs),1);
    for j = 1:length(subdirs)
        subdirs_num(j) = str2num(subdirs{j});
    end
    nb_subdir = length(subdirs_num);
    
   
    for j = 1:nb_subdir
        audio_input_dir = strcat(current_subdir,'\',num2str(subdirs_num(j)));

        mixed_audio_input_dir = strcat(audio_input_dir,'\',mixed_audio_input_name);
        clean_audio_input_dir = strcat(audio_input_dir,'\',clean_audio_input_name);

        mbstoi_file = load(strcat(audio_input_dir,'\','mbstoi.mat'));

        mbstoi = mbstoi_file.mbstoi;

        [y, fs_mixed] = audioread(mixed_audio_input_dir);
        [x, fs_clean] = audioread(clean_audio_input_dir);


        if fs_mixed ~= fs_clean
            error('mixed and clean sampling rates do not match');
        end

        y = reshape(zscore(y(:)),size(y,1),size(y,2));
        x = reshape(zscore(x(:)),size(x,1),size(x,2));


        y = resample(y,fs_preprocessing,fs_mixed);
        x = resample(x,fs_preprocessing,fs_clean);



        yl = y(:,1);
        yr = y(:,2);

        xl = x;
        xr = x;

        [xl,xr,yl,yr] = remove_silent_frames(xl,xr,yl,yr,dyn_range,N_frame,N_frame/2);

        xl_hat     	= stdft(xl, N_frame, N_frame/2, K);
        xr_hat     	= stdft(xr, N_frame, N_frame/2, K);
        yl_hat     	= stdft(yl, N_frame, N_frame/2, K);
        yr_hat     	= stdft(yr, N_frame, N_frame/2, K);

        xl_hat       = xl_hat(:, 1:(K/2+1)).';
        xr_hat       = xr_hat(:, 1:(K/2+1)).';
        yl_hat       = yl_hat(:, 1:(K/2+1)).';
        yr_hat       = yr_hat(:, 1:(K/2+1)).';


        xl_hat_mag = abs(xl_hat);
        xr_hat_mag = abs(xr_hat);
        yl_hat_mag = abs(yl_hat);
        yr_hat_mag = abs(yr_hat);

        xl_hat_mag = xl_hat_mag(4:end-20,:);
        xr_hat_mag = xr_hat_mag(4:end-20,:);
        yl_hat_mag = yl_hat_mag(4:end-20,:);
        yr_hat_mag = yr_hat_mag(4:end-20,:);


        xl_hat_angle = angle(xl_hat);
        xr_hat_angle = angle(xr_hat);
        yl_hat_angle = angle(yl_hat);
        yr_hat_angle = angle(yr_hat);

        xl_mfcc = mfcc(xl_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',13);
        xr_mfcc = mfcc(xr_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',13);
        yl_mfcc = mfcc(yl_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',13);
        yr_mfcc = mfcc(yr_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',13);
            
            

        mkdir(strcat(audio_output_base_dir,'\',data_split(i),'\',num2str(j)));
        save(strcat(audio_output_base_dir,'\',data_split(i),'\',num2str(j),'\','mfcc.mat'),'xl_mfcc','xr_mfcc','yl_mfcc','yr_mfcc','-v6');

        save(strcat(audio_output_base_dir,'\',data_split(i),'\',num2str(j),'\','stft_mag.mat'),'xl_hat_mag','xr_hat_mag','yl_hat_mag','yr_hat_mag','-v6');
        save(strcat(audio_output_base_dir,'\',data_split(i),'\',num2str(j),'\','stft_phase.mat'),'xl_hat_angle','xr_hat_angle','yl_hat_angle','yr_hat_angle','-v6');
        save(strcat(audio_output_base_dir,'\',data_split(i),'\',num2str(j),'\','mbstoi.mat'),'mbstoi','-v6');

            
         
            
    end
    save(strcat(audio_output_base_dir,'\',data_split(i),'\','H_trunc.mat'),'H_trunc');     
            
           
        
        

        
        
        
        
 
end




