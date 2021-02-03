clear all

addpath(genpath('MBSTOI'));
addpath(genpath('submodules'));
addpath(genpath('other_dependencies'));
addpath(genpath('RIR-Generator-master'));



noise_file_type = '.mat';
noise_base_dir = 'E:\FYP\MATLAB\NOISEX-92';

noise_files = dir(strcat(noise_base_dir,'\','*',noise_file_type));

noise_names = {noise_files.name};

anechoic_base_dir = 'E:\FYP\MATLAB\Timit (subset) Speech dataset';
speaker = 'FDAW0';
sentence = 'SA1';
anechoic_file_type = '.wav';

anechoic_dir = strcat(anechoic_base_dir,'\',speaker,'\',sentence,anechoic_file_type);
[anechoic_speech, fs_anechoic] = audioread(anechoic_dir);
audio_out_base_dir = 'E:\FYP\Audio_Preprocessing_Input\ToyExample';
mixed_audio_out_file_name = 'mixed.wav';
clean_audio_out_file_name = 'clean.wav';

hrir_base_dir = 'E:\IP\Kayser2009\HRIR_database_wav\hrir\anechoic';

hrir_name = 'anechoic_distcm_80_el_0_az_0';

hrir_file_type = '.wav';

hrir_dir = strcat(hrir_base_dir,'\',hrir_name,hrir_file_type);

[hrir, fs_hrir] = audioread(hrir_dir);
if fs_hrir ~= fs_anechoic
    hrir = resample(hrir,fs_anechoic,fs_hrir);
end
hrir = hrir(:,[3,4,7,8]);



speech_out = fftfilt(hrir,anechoic_speech);

for i =1:size(noise_names,2)
    disp(i);
    noise_dir = strcat(noise_base_dir,'\',noise_names{i});
    audio_out_dir = strcat(audio_out_base_dir,'\',int2str(i));
    

    noise = load(strcat(noise_base_dir,'\',noise_names{i})).(erase(noise_names{i},'.mat'));
    fs_noise = 19980;
    noise = resample(noise,fs_anechoic,fs_noise);
    
    noise_out = fftfilt(hrir,noise);
    
    noise_out = noise_out(1:size(speech_out,1),:);
    
    
    count = 1;
    for snr_level = -30:3:30
        disp(snr_level);
        mixed_audio_out_snr_dir = strcat(audio_out_dir,'\',int2str(count),'\',mixed_audio_out_file_name);
        s_db=v_activlev(speech_out,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*speech_out;
        s_db=v_activlev(mod_sound,16000,'d');  % speech level in dB
        mod_sound = 10^((-(s_db))/20)*mod_sound;
        s_db = v_activlev(mod_sound,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(noise_out.^2)/size(noise_out,1)));
        mixed_out = mod_sound + 10^(-(snr_level + n_db)/20)*noise_out;
        
        clean_audio_out_snr_dir = strcat(audio_out_dir,'\',int2str(count),'\',clean_audio_out_file_name);
        v_writewav(speech_out,16000,clean_audio_out_snr_dir,[],[],[],[]);
        v_writewav(mixed_out,16000,mixed_audio_out_snr_dir,[],[],[],[]);
        count = count + 1;
    end
    
  
    
end
noise_names = noise_names.';
save(strcat(audio_out_base_dir,'\','dataset_description.mat'),'noise_names','hrir_dir','anechoic_dir');
    
    
    
