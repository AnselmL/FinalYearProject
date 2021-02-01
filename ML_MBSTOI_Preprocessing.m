clear all;


addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));

%specify audio input directory
audio_input_base_dir = 'E:\FYP\Audio_Preprocessing_Input\ToyExample';
audio_input_dir_list = strrep(audio_input_base_dir,'\','/');
mixed_audio_input_name = 'mixed.wav';
clean_audio_input_name = 'clean.wav';
%Also specify features output directory
feature_output_base_dir = 'E:\FYP\Feature_Preprocessing_Output\ToyExample';
feature_output_name = 'features.mat';


%specifications for fft
fs_preprocessing        = 10000;    % Sampling rate
N_frame                 = 256;    	% Window support [samples].
K                       = 512;     	% FFT size [samples].
K_phase                 = 256;      % FFT size for phase features
dyn_range               = 40;      	% Speech dynamic range [dB].
intermediate_frame_size = 4224;



%calculating MFCC filter-band edges
fmin = 100;
fmax = 4500;
mel_filter_num = 34;

fmin_mel = hz2mel(fmin);
fmax_mel = hz2mel(fmax);

band_edges_mels = linspace(fmin_mel, fmax_mel, mel_filter_num+2);

band_edges_freqs = mel2hz(band_edges_mels);



dir_list=dir(audio_input_dir_list);
dirflags = [dir_list.isdir];
subdir = dir_list(dirflags);
subdir(ismember( {subdir.name}, {'.', '..'})) = [];  %remove . and ..
nb_subdir = length(subdir);
nb_snr = 21;


for i = 1:nb_subdir
    disp(i)
    for j = 1:nb_snr
        mixed_audio_input_dir = strcat(audio_input_base_dir,'\',int2str(i),'\',int2str(j),'\',mixed_audio_input_name);
        clean_audio_input_dir = strcat(audio_input_base_dir,'\',int2str(i),'\',int2str(j),'\',clean_audio_input_name);
        
        [y, fs_mixed] = audioread(mixed_audio_input_dir);
        [x, fs_clean] = audioread(clean_audio_input_dir);
        
        if fs_mixed ~= fs_clean
            error('mixed and clean sampling rates do not match');
        end
        
        y = resample(y,fs_preprocessing,fs_mixed);
        x = resample(x,fs_preprocessing,fs_clean);
        

        
        yl = (y(:,1) + y(:,3))/2;
        yr = (y(:,2) + y(:,4))/2;
        

        
        xl = (x(:,1) + x(:,3))/2;
        xr = (x(:,2) + x(:,4))/2;
        
        
        
        [xl,xr,yl,yr] = remove_silent_frames(xl,xr,yl,yr,dyn_range,N_frame,N_frame/2);
        
        
        
        %zero-mean, unit-variance normalization
        yl = reshape(zscore(yl(:)),size(yl,1),size(yl,2));
        yr = reshape(zscore(yr(:)),size(yr,1),size(yr,2));
        xl = reshape(zscore(xl(:)),size(xl,1),size(xl,2));
        xr = reshape(zscore(xr(:)),size(xr,1),size(xr,2));
        
        nb_intermediate_frames = floor(length(xl)/intermediate_frame_size); %doesn't matter which one since they all have the same length
        
        for k = 1:nb_intermediate_frames
            start_ind = (k-1)*intermediate_frame_size + 1;
            end_ind = k*intermediate_frame_size;
            
            mbstoi_val = mbstoi(xl(start_ind:end_ind),xr(start_ind:end_ind),yl(start_ind:end_ind),yr(start_ind:end_ind),fs_preprocessing);
            
            xl_hat     	= stdft(xl, N_frame, N_frame/2, K);
            xr_hat     	= stdft(xr, N_frame, N_frame/2, K);
            yl_hat     	= stdft(yl, N_frame, N_frame/2, K);
            yr_hat     	= stdft(yr, N_frame, N_frame/2, K);
            
            xl_hat = xl_hat.';
            xr_hat = xr_hat.';
            yl_hat = yl_hat.';
            yr_hat = yr_hat.';
            
            xl_mfcc = mfcc(xl_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',8);
            xr_mfcc = mfcc(xr_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',8);
            yl_mfcc = mfcc(yl_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',8);
            yr_mfcc = mfcc(yr_hat,fs_preprocessing,'LogEnergy','Append','BandEdges',band_edges_freqs,'NumCoeffs',8);
            
            mfcc_corr_matrix_left = corr(xl_mfcc,yl_mfcc);
            mfcc_corr_matrix_right = corr(xr_mfcc,yr_mfcc);
            
            mfcc_features_left = diag(mfcc_corr_matrix_left);
            mfcc_features_right = diag(mfcc_corr_matrix_right);
            
            
            xl_hat_phase     	= stdft(xl, N_frame, N_frame/2, K_phase).';
            xr_hat_phase     	= stdft(xr, N_frame, N_frame/2, K_phase).';
            yl_hat_phase     	= stdft(yl, N_frame, N_frame/2, K_phase).';
            yr_hat_phase     	= stdft(yr, N_frame, N_frame/2, K_phase).';
            
            xl_hat_phase       = xl_hat_phase(1:(K_phase/2+1),:);
            xr_hat_phase       = xr_hat_phase(1:(K_phase/2+1),:);
            yl_hat_phase       = yl_hat_phase(1:(K_phase/2+1),:);
            yr_hat_phase       = yr_hat_phase(1:(K_phase/2+1),:);
            
            phase_xl_pre = angle(xl_hat_phase);
            phase_xr_pre = angle(xr_hat_phase);
            phase_yl_pre = angle(yl_hat_phase);
            phase_yr_pre = angle(yr_hat_phase);
            %{
            %phase_xl_mod = phase_xl/phase_xl(51,:);
            xl_zero_ind = find(~phase_xl_pre(26,:));
            xr_zero_ind = find(~phase_xr_pre(26,:));
            yl_zero_ind = find(~phase_yl_pre(26,:));
            yr_zero_ind = find(~phase_yr_pre(26,:));
            
            
            if ~isempty(xl_zero_ind)
                for ind = 1:size(xl_zero_ind)
                    
                end
            %}
            
            phase_xl = bsxfun(@rdivide, phase_xl_pre, phase_xl_pre(27,:));
            phase_xr = bsxfun(@rdivide, phase_xr_pre,hcj phase_xr_pre(27,:));
            phase_yl = bsxfun(@rdivide, phase_yl_pre, phase_yl_pre(27,:));
            phase_yr = bsxfun(@rdivide, phase_yr_pre, phase_yr_pre(27,:));
            
            phase_xl = phase_xl(2:26,:);
            if any(isnan(phase_xl))
                error('nan in phase_xl')
            end
            phase_xr = phase_xr(2:26,:);
            if any(isnan(phase_xr))
                error('nan in phase_xr');
            end
            phase_yl = phase_yl(2:26,:);
            if any(isnan(phase_yl))
                error('nan in phase_yl')
            end
            phase_yr = phase_yr(2:26,:);
            if any(isnan(phase_yr))
                error('nan in phase_yr')
            end
            
            phase_diff_clean = phase_xr - phase_xl;
            
            phase_diff_noisy = phase_yr - phase_yl;
            
            phase_diff_corr_matrix = corr(phase_diff_clean.',phase_diff_noisy.');
            phase_features = diag(phase_diff_corr_matrix);
            %{
            phase_corr_matrix_left = corr(phase_xl.',phase_yl.');
            phase_corr_matrix_right = corr(phase_xr.',phase_yr.');
            phase_features_left = diag(phase_corr_matrix_left);
            phase_features_right = diag(phase_corr_matrix_right);
            %}
            
            if mbstoi_val < 0
                disp('mbstoi less than 0');
            end
            if isnan(mbstoi_val)
                disp('mbstoi is nan');
            end
            features = cat(1,mfcc_features_left, mfcc_features_right,phase_features);
            label = mbstoi_val;
            nfeatures = size(features,1);
            feature_output_dir = strcat(feature_output_base_dir,'\',int2str(i),'\',int2str(j),'\',feature_output_name);
            save(feature_output_dir,'features','label','nfeatures');
            
            
            
            
            
         
            
        end
            
            
           
        
        

        
        
        
        
        
    end
end
%read audio file



%resample audio to 10Khz

%remove silent frames ->equivalent to taking VAD in real-time

%compute MBSTOI for 4096 intervals of pure speech

%take stft

%compute mel frequency cepstrum coefficients (MFCC)

%compute phase information

%use 100 - 1500 Khz





