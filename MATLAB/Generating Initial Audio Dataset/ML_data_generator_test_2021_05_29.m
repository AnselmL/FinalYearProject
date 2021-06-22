clear all;
addpath(genpath('D:\FYP\MATLAB\MBSTOI'));
addpath(genpath('D:\FYP\MATLAB\submodules'));
addpath(genpath('D:\FYP\MATLAB\other_dependencies'));
addpath(genpath('D:\FYP\MATLAB\RIR-Generator-master'));
addpath(genpath('D:\FYP\MATLAB\ANF-Generator-master'));


%{
[x, fs_x] = audioread('anechoic_speech.wav');
y = rand(size(x,1),1);

[z,p,fs_o] = v_addnoise(x,fs_x,10,'dxopEkn',y,fs_x);
%}

TIMIT_wav_dirs = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_test;
measured_BRIR_wav_dirs = load('BRIR_wav_dirs_2021_05_29').BRIR_wav_dirs;

base_dir = 'D:\FYP\MATLAB';
%HRTF_base_dir = 'D:\T5-Backup\FYP\MATLAB\HRTF\TEST';

out_base_dir = 'D:\FYP\feature_preprocessing_input\directional_2021_05_28\TEST';

%out_base_unnorm_dir = 'E:\FYP\ML_data_dir\simulated\unnorm\1-fold';

noise_wav_dirs = load('TUT_wav_dirs_testing').TUT_wav_dirs;

mic_pos_Aachen_KEMAR = [-0.0700,0,0;
                    0.0700,0,0];

mic_pos_NF_KOELN = [-0.0875,0,0;
                    0.0875,0,0];
                
mic_pos_Kayser_bte = [-0.08200,-0.02105,0.03635;
                      0.08200,-0.02105,0.03635];
                  
mic_pos_eBrIRD_bte = [-0.0800,-0.0200,0.0250;
                     0.0800,-0.0200,0.0250];
                 
mic_pos_AIR = [-0.0850,0,0;
               0.0850,0,0];


TIMIT_wav_pool = [];

noise_index_pool = [];
SNR_diff_index_pool = [];
SNR_direc_index_pool = [];
measured_BRIR_pool = [];


direc_pool = [];
direc_type_pool = [];


TIMIT_train_file_count = 1;

nb_folds = 3;


SNR_diff = -9:3:30;
SNR_direc = -6:3:15;
direc_type = [0,1]; %0 -> noise source, 1 -> speech
direc_on = [0,1,2]; %0 -> no directional interfer, 1 -> directional interfer, 2 -> 2 directional interferers


direc_type_index = cell(1,2);
SNR_direc_index = cell(1,2);
brir_out_dir_direc = cell(1,2);
BRIR_direc = cell(1,2);
noise_direc_dir = cell(1,2);
TIMIT_direc_dir = cell(1,2);

file_counter = 1;
for fold = 1:nb_folds
    if isempty(TIMIT_wav_pool)
        TIMIT_wav_pool = 1:size(TIMIT_wav_dirs,1);
    end

    while(~isempty(TIMIT_wav_pool))
        %clear BRIR
        %clear BRIR_direc
        clear BRIR_temp
        clear BRIR_direc_temp
        if isempty(measured_BRIR_pool)
            measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
        end

        if isempty(noise_index_pool)
            noise_index_pool = 1:size(noise_wav_dirs,1);
        end
        if isempty(SNR_diff_index_pool)    
            SNR_diff_index_pool = 1:length(SNR_diff);
        end
        if isempty(SNR_direc_index_pool)    
            SNR_direc_index_pool = 1:length(SNR_direc);
        end

        if isempty(direc_pool)
            direc_pool = 1:3;
        end
        if isempty(direc_type_pool)
            direc_type_pool = 1:2;
        end



        TIMIT_wav_index = TIMIT_wav_pool(randperm(size(TIMIT_wav_pool,2),1));
        TIMIT_wav_pool=TIMIT_wav_pool(TIMIT_wav_pool~=TIMIT_wav_index);
        anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_wav_index,1};

        [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        %{
        if ~isempty(measured_BRIR_pool)
            measured_vs_simulated_index = measured_vs_simulated_pool(randperm(size(measured_vs_simulated_pool,2),1));
            measured_vs_simulated_pool=measured_vs_simulated_pool(measured_vs_simulated_pool~=measured_vs_simulated_index);
        else
            measured_vs_simulated_index = 2;
        end
        %}
        %{
        %fetching speech impulse response
        if  measured_vs_simulated_index == 1
            measured_BRIR_index = measured_BRIR_pool(randperm(size(measured_BRIR_pool,2),1));
            %measured_BRIR_index = nearest(rand*size(measured_BRIR_pool,2)+1);
            measured_BRIR_pool=measured_BRIR_pool(measured_BRIR_pool~=measured_BRIR_index);
            BRIR_dir = measured_BRIR_wav_dirs{measured_BRIR_index,1};
            [HRTF, fs_brir] = audioread(BRIR_dir);
            if fs_brir ~= fs_speech
                fs_brir = resample(fs_brir,fs_speech,fs_brir);
            end
            %checking type of measured BRIR
            Kayser_strfind = strfind(BRIR_dir,'Kayser');
            AIR_strfind = strfind(BRIR_dir,'Aachen');
            eBrIRD_strfind = strfind(BRIR_dir,'eBrIRD');
            if ~isempty(Kayser_strfind)
                Kayser_office_I_strfind = strfind(BRIR_dir,'office_I');
                if isempty(Kayser_office_I_strfind)    
                    Kayser_eBrIRD_IR_type_index = Kayser_eBrIRD_IR_type_pool(randperm(size(Kayser_eBrIRD_IR_type_pool,2),1));
                    Kayser_eBrIRD_IR_type_pool = Kayser_eBrIRD_IR_type_pool(Kayser_eBrIRD_IR_type_pool~= Kayser_eBrIRD_IR_type_index);
                else
                    Kayser_eBrIRD_IR_type_index = 3;
                end
                if Kayser_eBrIRD_IR_type_index == 1
                    %BTE
                    BRIR(:,1) = (HRTF(:,3) + HRTF(:,7))/2;
                    BRIR(:,2) = (HRTF(:,4) + HRTF(:,8))/2;

                    mic_position = mic_pos_Kayser_bte;
                elseif Kayser_eBrIRD_IR_type_index == 2
                    %IR
                    BRIR(:,1) = HRTF(:,1);
                    BRIR(:,2) = HRTF(:,2);

                    mic_position = mic_pos_Kayser_ie;
                elseif Kayser_eBrIRD_IR_type_index == 3
                    BRIR(:,1) = (HRTF(:,1) + HRTF(:,5))/2;
                    BRIR(:,2) = (HRTF(:,2) + HRTF(:,6))/2;
                else
                    error('something went wrong')
                end
                %{
                %data augmentation
                speech_type = nearest(rand(1));
                if speech_type == 0
                    %simply TIMIT
                elseif speech_type == 1
                    %convolve TIMIT with Kayser anechoic frontal 0.8m impulse
                    %response
                end
                %}

            elseif ~isempty(AIR_strfind)
                mic_position = mic_pos_AIR;
                BRIR = HRTF;
            elseif ~isempty(eBrIRD_strfind)
                Kayser_eBrIRD_IR_type_index = Kayser_eBrIRD_IR_type_pool(randperm(size(Kayser_eBrIRD_IR_type_pool,2),1));
                Kayser_eBrIRD_IR_type_pool = Kayser_eBrIRD_IR_type_pool(Kayser_eBrIRD_IR_type_pool~= Kayser_eBrIRD_IR_type_index);
                if Kayser_eBrIRD_IR_type_index == 1
                    %BTE
                    HRTF = HRTF(:,3:6);
                    BRIR(:,1) = (HRTF(:,1) + HRTF(:,3))/2; %left brir
                    BRIR(:,2) = (HRTF(:,2) + HRTF(:,4))/2; %right brir
                    mic_position = mic_pos_eBrIRD_bte;
                elseif Kayser_eBrIRD_IR_type_index == 2
                    %IE
                    BRIR = HRTF(:,1:2);
                    mic_position = mic_pos_eBrIRD_ie;
                else
                    error('something went wrong')
                end
                %{
                %data augmentation
                speech_type = nearest(rand(1));
                if speech_type == 0
                    %simply TIMIT
                elseif speech_type == 1
                    %convolve TIMIT with Kayser anechoic frontal 0.8m impulse
                    %response
                end
                %}
            else
                error('something went wrong')
            end

        %}
        %simulated
        %mic_position_type = nearest(rand(6));
        
        measured_BRIR_index = measured_BRIR_pool(randperm(size(measured_BRIR_pool,2),1));
        %measured_BRIR_index = nearest(rand*size(measured_BRIR_pool,2)+1);
        measured_BRIR_pool=measured_BRIR_pool(measured_BRIR_pool~=measured_BRIR_index);
        BRIR_dir = measured_BRIR_wav_dirs{measured_BRIR_index,1};

        [BRIR, fs_BRIR] = audioread(BRIR_dir);
        
        BRIR = resample(BRIR,fs_speech,fs_BRIR);
        
        
        

        if size(BRIR,2) > 2 && contains(BRIR_dir,'Kayser') && ~contains(BRIR_dir,'office_I')
            BRIR_temp(:,1) = (BRIR(:,3) + BRIR(:,7))/2;
            BRIR_temp(:,2) = (BRIR(:,4) + BRIR(:,8))/2;
            BRIR = BRIR_temp;
        elseif size(BRIR,2) > 2 && contains(BRIR_dir,'office_I')
            BRIR_temp(:,1) = (BRIR(:,1) + BRIR(:,5))/2;
            BRIR_temp(:,2) = (BRIR(:,2) + BRIR(:,6))/2;
            BRIR = BRIR_temp;
        elseif size(BRIR,2) > 2 && contains(BRIR_dir,'eBrIRD')
            BRIR_temp(:,1) = (BRIR(:,3) + BRIR(:,5))/2;
            BRIR_temp(:,2) = (BRIR(:,4) + BRIR(:,6))/2;
            BRIR = BRIR_temp;
        end



        speech_binaural = fftfilt(BRIR, anechoic_speech);

        noise_index = noise_index_pool(randperm(size(noise_index_pool,2),1));
        noise_index_pool=noise_index_pool(noise_index_pool~=noise_index);
        noise_dir = noise_wav_dirs{noise_index};
        [noise_wav,fs_noise] = audioread(noise_dir);
        if fs_noise ~= fs_speech
            noise_wav = resample(noise_wav,fs_speech,fs_noise);
            fs_noise = fs_speech;
        end
        noise_mono = zeros(size(noise_wav,1),1);

        for chan = 1:size(noise_wav,2)
            noise_mono = noise_mono + noise_wav(:,chan);
        end
        %noise_mono = noise_mono/max(max(abs(noise_mono)));
        if 2*size(anechoic_speech,1) > length(noise_mono)
            noise_mono_temp = [noise_mono;noise_mono];
            noise_mono = noise_mono_temp;
        end
                   
       if contains(BRIR_dir,'AIR')
           mic_position = mic_pos_AIR;
       elseif contains(BRIR_dir,'Aachen_KEMAR')
           mic_position = mic_pos_Aachen_KEMAR;
       elseif contains(BRIR_dir,'Kayser')
           mic_position = mic_pos_Kayser_bte;
       elseif contains(BRIR_dir,'eBrIRD')
           mic_position = mic_pos_eBrIRD_bte;
       elseif contains(BRIR_dir,'NF_KOELN')
           mic_position = mic_pos_NF_KOELN;
       else
           error('unrecognized mic position')
       end
        
        diffuse_noise = gen_isotropic_noise_2020_12_17(noise_mono,fs_noise,mic_position,size(anechoic_speech,1));
        if size(diffuse_noise,1) ~= size(speech_binaural,1)
            diffuse_noise = diffuse_noise(1:size(speech_binaural,1),1);
        end

        fs = fs_speech;








        s_db=v_activlev(speech_binaural,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*speech_binaural;
        s_db=v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*mod_binaural_speech;
        s_db = v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));

        SNR_diff_index = SNR_diff_index_pool(randperm(length(SNR_diff_index_pool),1));
        SNR_diff_index_pool=SNR_diff_index_pool(SNR_diff_index_pool~=SNR_diff_index);

        binaural_out = mod_binaural_speech + 10^(-(SNR_diff(SNR_diff_index) + n_db)/20)*diffuse_noise;


        direc_index = direc_pool(randperm(length(direc_pool),1));
        direc_pool =direc_pool(direc_pool~=direc_index);
        
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            direc_type_index{1} = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index{1});
            
            SNR_direc_index{1} = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index{1});
            
            SNR_direc_index{2} = randi(length(SNR_direc));

            subdir_ind = strfind(BRIR_dir,'\');
            last_slash = subdir_ind(end);
            direc_end = BRIR_dir(last_slash+1:end);
            dir_temp = BRIR_dir(1:last_slash);
            wav_files=dir(strcat(dir_temp,'*','.wav'));
            wav_files = {wav_files.name};
            wav_files(ismember(wav_files,direc_end)) = [];
            if contains(BRIR_dir,'eBrIRD')
                if ~contains(BRIR_dir,'Anechoic')
                    az_ind = strfind(BRIR_dir,'az_');
                    wav_ind = strfind(BRIR_dir,'.wav');
                    az_shortstring = BRIR_dir(az_ind:wav_ind-1);
                    wav_files(~contains(wav_files,az_shortstring)) = [];
                end
            end
            
            direc_ind = randperm(length(wav_files),2);
            brir_out_dir_direc{1} = strcat(dir_temp,wav_files{direc_ind(1)});

            [BRIR_direc{1},fs_BRIR_direc(1)] = audioread(brir_out_dir_direc{1});
            
            BRIR_direc{1} = resample(BRIR_direc{1},fs_speech,fs_BRIR_direc(1));
            
            if size(BRIR_direc{1},2) > 2 && contains(brir_out_dir_direc{1},'Kayser') && ~contains(brir_out_dir_direc{1},'office_I')
                BRIR_direc_temp(:,1) = (BRIR_direc{1}(:,3) + BRIR_direc{1}(:,7))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{1}(:,4) + BRIR_direc{1}(:,8))/2;
                BRIR_direc{1} = BRIR_direc_temp;
            elseif size(BRIR_direc{1},2) > 2 && contains(brir_out_dir_direc{1},'office_I')
                BRIR_direc_temp(:,1) = (BRIR_direc{1}(:,1) + BRIR_direc{1}(:,5))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{1}(:,2) + BRIR_direc{1}(:,6))/2;
                BRIR_direc{1} = BRIR_direc_temp;
            elseif size(BRIR_direc{1},2) > 2 && contains(brir_out_dir_direc{1},'eBrIRD')
                BRIR_direc_temp(:,1) = (BRIR_direc{1}(:,3) + BRIR_direc{1}(:,5))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{1}(:,4) + BRIR_direc{1}(:,6))/2;
                BRIR_direc{1} = BRIR_direc_temp;
            end

                
            if direc_on(direc_index) == 2
                direc_type_index{2} = randi(2);
                
                brir_out_dir_direc{2} = strcat(dir_temp,wav_files{direc_ind(2)});

                [BRIR_direc{2},fs_BRIR_direc(2)] = audioread(brir_out_dir_direc{2});


                BRIR_direc{2} = resample(BRIR_direc{2},fs_speech,fs_BRIR_direc(2));
                
            if size(BRIR_direc{2},2) > 2 && contains(brir_out_dir_direc{2},'Kayser') && ~contains(brir_out_dir_direc{2},'office_I')
                BRIR_direc_temp(:,1) = (BRIR_direc{2}(:,3) + BRIR_direc{2}(:,7))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{2}(:,4) + BRIR_direc{2}(:,8))/2;
                BRIR_direc{2} = BRIR_direc_temp;
            elseif size(BRIR_direc{2},2) > 2 && contains(brir_out_dir_direc{2},'office_I')
                BRIR_direc_temp(:,1) = (BRIR_direc{2}(:,1) + BRIR_direc{2}(:,5))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{2}(:,2) + BRIR_direc{2}(:,6))/2;
                BRIR_direc{2} = BRIR_direc_temp;
            elseif size(BRIR_direc{2},2) > 2 && contains(brir_out_dir_direc{2},'eBrIRD')
                BRIR_direc_temp(:,1) = (BRIR_direc{2}(:,3) + BRIR_direc{2}(:,5))/2;
                BRIR_direc_temp(:,2) = (BRIR_direc{2}(:,4) + BRIR_direc{2}(:,6))/2;
                BRIR_direc{2} = BRIR_direc_temp;
            end

            end
            
            if direc_type(direc_type_index{1}) == 0
                direc_file_index = randi(size(noise_wav_dirs,1));
                noise_direc_dir{1} = noise_wav_dirs{direc_file_index};
                [direc_full, fs_direc] = audioread(noise_direc_dir{1});
                if size(direc_full,2) > 1
                    direc_full = mean(direc_full,2);
                end
                if fs_direc ~= fs
                    direc_full = resample(direc_full,fs,fs_direc);
                end
                [~,I] = max(direc_full);
                overshoot = round(size(speech_binaural,1)/2);
                out_length = size(speech_binaural,1);
                if I >= out_length/2
                    while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                elseif I <= out_length/2
                    while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                end

                direc_binaural{1} = fftfilt(BRIR_direc{1},direc);




                nd_db = mean(pow2db(sum(direc_binaural{1}.^2)/size(direc_binaural{1},1)));
                direc_binaural{1} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
            elseif direc_type(direc_type_index{1}) == 1
                TIMIT_direc_dir{1} = randsample(TIMIT_wav_dirs,1);
                [direc,fs_direc] = audioread(TIMIT_direc_dir{1}{1});
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                direc_binaural{1} = fftfilt(BRIR_direc{1},direc);
                sd_db = v_activlev(direc_binaural{1},16000,'d');
                direc_binaural{1} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
            end
            binaural_out = binaural_out + direc_binaural{1};



            if direc_on(direc_index) == 2
                direc_type_index{2} = randi(2);

                if direc_type(direc_type_index{2}) == 0
                    direc_file_index = randi(size(noise_wav_dirs,1));
                    noise_direc_dir{2} = noise_wav_dirs{direc_file_index};
                    [direc_full, fs_direc] = audioread(noise_direc_dir{2});
                    if size(direc_full,2) > 1
                        direc_full = mean(direc_full,2);
                    end
                    if fs_direc ~= fs
                        direc_full = resample(direc_full,fs,fs_direc);
                    end
                    [~,I] = max(direc_full);
                    overshoot = round(size(speech_binaural,1)/2);
                    out_length = size(speech_binaural,1);
                    if I >= out_length/2
                        while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                            overshoot = round(overshoot/2);
                            if overshoot < 5
                                overshoot = 0;
                            end
                        end
                        direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                    elseif I <= out_length/2
                        while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                            overshoot = round(overshoot/2);
                            if overshoot < 5
                                overshoot = 0;
                            end
                        end
                        direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                    end

                    direc_binaural{2} = fftfilt(BRIR_direc{2},direc);

                    nd_db = mean(pow2db(sum(direc_binaural{2}.^2)/size(direc_binaural{2},1)));
                    direc_binaural{2} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
                elseif direc_type(direc_type_index{2}) == 1
                    TIMIT_direc_dir{2} = randsample(TIMIT_wav_dirs,1);
                    [direc,fs_direc] = audioread(TIMIT_direc_dir{2}{1});
                    if fs_direc ~= fs
                        direc = resample(direc,fs,fs_direc);
                    end
                    if length(direc) > size(speech_binaural,1)
                        direc = direc(1:size(speech_binaural,1),1);
                    elseif length(direc) < size(speech_binaural,1)
                        direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                    end
                    direc_binaural{2} = fftfilt(BRIR_direc{2},direc);
                    sd_db = v_activlev(direc_binaural{2},16000,'d');
                    direc_binaural{2} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
                end

                binaural_out = binaural_out + direc_binaural{2};
            end
            
            
        end


                
        %now in all events I have two BRIR_direc
        %here now I need to pull either one or two wav files (noise or
        %speech) and then perform the required convolutions
        %after that I can add these normally

            

            
            
            
            
            
        %{
        if direc_on(direc_index) == 1
            direc_type_index = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index);
            
            SNR_direc_index = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index);
            
            Kayser_strfind = strfind(BRIR_dir,'Kayser');
            AIR_strfind = strfind(BRIR_dir,'Aachen');
            eBrIRD_strfind = strfind(BRIR_dir,'eBrIRD');
            
            if ~isempty(eBrIRD_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                second_2last_slash = subdir_ind(end -1);
                direc_end = BRIR_dir(last_slash:end);
                cur_subdir = BRIR_dir(second_2last_slash + 1:last_slash - 1);
                dir_temp = BRIR_dir(1:second_2last_slash);
                dir_list=dir(dir_temp);
                dirflags = [dir_list.isdir];
                subdir = dir_list(dirflags);
                subdir(ismember( {subdir.name}, {'.', '..'})) = [];  %remove . and ..
                %nb_subdir = length(subdir);
                subdirs = {subdir.name};
                count = 1;
                subdirs(ismember(subdirs,cur_subdir)) = [];

                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,new_subdir,direc_end);
                [BRIR_direc_temp, fs_brir] = audioread(BRIR_dir);
                BRIR_direc_temp = resample(BRIR_direc_temp,fs,fs_brir);
                
                BRIR_direc_temp = BRIR_direc_temp(:,3:6);
                BRIR_direc(:,1) = (BRIR_direc_temp(:,1) + BRIR_direc_temp(:,3))/2; %left brir
                BRIR_direc(:,2) = (BRIR_direc_temp(:,2) + BRIR_direc_temp(:,4))/2; %right brir
            elseif ~isempty(AIR_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                direc_end = BRIR_dir(last_slash+1:end);
                dir_temp = BRIR_dir(1:last_slash);
                dir_list=dir(dir_temp);
                dir_list(ismember( {dir_list.name}, {'.', '..'})) = [];
                subdirs = {dir_list.name};
                %dirflags = [dir_list.isdir];
                %subdir = dir_list(dirflags);
                %subdir(ismember( {subdir.name}, {'.', '..'})) = [];
                %subdirs = {subdir.name};
                first_1_temp = strfind(BRIR_dir,'1');
                identifier = BRIR_dir(last_slash + 1:first_1_temp - 2);
                i= 1;
                while(i <= length(subdirs))
                    if ~contains(subdirs{i},identifier)
                        subdirs(i) = [];
                    else
                        i = i + 1;
                    end
                end
                %subdirs{contains(subdirs,identifier)} = [];
                subdirs(ismember(subdirs,direc_end)) = [];
                %{
                count = 1;
                for i = 1:length(subdirs)
                    if ~strcmp(subdirs{i},direc_end)      
                        subdirs_temp{count} = subdirs{i};
                        count = count + 1;
                    end
                end
                subdirs = subdirs_temp;
                %}
                %subdirs{subdirs == direc_end} = [];
                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,'\',new_subdir);
                [BRIR_direc, fs_brir] = audioread(BRIR_dir);
                
                BRIR_direc = resample(BRIR_direc,fs,fs_brir);
            elseif ~isempty(Kayser_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                direc_end = BRIR_dir(last_slash+1:end);
                dir_temp = BRIR_dir(1:last_slash);
                dir_list=dir(dir_temp);
                dir_list(ismember( {dir_list.name}, {'.', '..'})) = [];
                subdirs = {dir_list.name};
                
                subdirs(ismember(subdirs,direc_end)) = [];
                %dirflags = [dir_list.isdir];
                %subdir = dir_list(dirflags);

                %subdirs = {subdir.name};
                %{
                subdirs_temp = cell(length(subdirs),1);
                count = 1;
                for i = 1:length(subdirs)
                    if ~strcmp(subdirs{i},direc_end)      
                        subdirs_temp{count} = subdirs{i};
                        count = count + 1;
                    end
                end
                subdirs = subdirs_temp;
                %}
                %subdirs{subdirs == direc_end} = [];
                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,'\',new_subdir);
                [BRIR_direc_temp, fs_brir] = audioread(BRIR_dir);
                BRIR_direc_temp = resample(BRIR_direc_temp,fs,fs_brir);
                if contains(direc_BRIR_dir,'office_I')
                    BRIR_direc(:,1) = (HRTF(:,1) + HRTF(:,5))/2;
                    BRIR_direc(:,2) = (HRTF(:,2) + HRTF(:,6))/2;
                else
                    BRIR_direc(:,1) = (BRIR_direc_temp(:,3) + BRIR_direc_temp(:,7))/2;
                    BRIR_direc(:,2) = (BRIR_direc_temp(:,4) + BRIR_direc_temp(:,8))/2;
                end
            end
            
            if direc_type(direc_type_index) == 0
                direc_file_index = floor(rand(1)*size(noise_wav_dirs,1) + 1);
                [direc, fs_direc] = audioread(noise_wav_dirs{direc_file_index});
                if size(direc,2) > 1
                    direc = mean(direc,2);
                end
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                direc = direc(1:size(speech_binaural,1),1);
                
                direc_binaural = fftfilt(BRIR_direc,direc);

                nd_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));
                direc_binaural = 10^((-(nd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
                elseif direc_type(direc_type_index) == 1
                TIMIT_direc_dir = 'blabla';
                while(~contains(TIMIT_direc_dir,out_data_subdir))
                    TIMIT_direc_dir = randsample(TIMIT_wav_dirs,1);
                    TIMIT_direc_dir = TIMIT_direc_dir{1};
                end
                [direc,fs_direc] = audioread(TIMIT_direc_dir);
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                direc_binaural = fftfilt(BRIR_direc,direc);
                sd_db = v_activlev(speech_binaural,16000,'d');
                direc_binaural = 10^((-(sd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
            end

            binaural_out = binaural_out + direc_binaural;
        end
        %}
            
        %here also have anechoic_hrtf_speech, where the speech has been
        %convolved with hrtf for mbstoi calculation
        
        %I have to make sure at this point that I have an hrtf saved in all
        %cases
        
        %anechoic_speech_hrtf = fftfilt(HRIR,anechoic_speech);
        
        
        binaural_out = reshape(zscore(binaural_out(:)),size(binaural_out,1),size(binaural_out,2));
        anechoic_speech = reshape(zscore(anechoic_speech(:)),size(anechoic_speech,1),size(anechoic_speech,2));
        %anechoic_speech_hrtf = reshape(zscore(anechoic_speech_hrtf(:)),size(anechoic_speech_hrtf,1),size(anechoic_speech_hrtf,2));
            
        
        xl = anechoic_speech;
        xr = anechoic_speech;
        %xl_hrtf = anechoic_speech_hrtf(:,1);
        %xr_hrtf = anechoic_speech_hrtf(:,2);

        yl = binaural_out(:,1);
        yr = binaural_out(:,2);
        %calculate both mbstoi for clean and hrtf anechoic speech
        mbstoi = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);
        %mbstoi_hrtf = mbstoi_intermediate(xl_hrtf, xr_hrtf, yl, yr, fs_speech);

        out_dir = strcat(out_base_dir,'\',num2str(file_counter));
        disp(out_dir);
        mkdir(out_dir);
        mbstoi_file_name = strcat(out_dir,'\','mbstoi.mat');
        %mbstoi_hrtf_file_name = strcat(out_dir,'\','mbstoi_hrtf.mat');
        save(mbstoi_file_name,'mbstoi');
        %save(mbstoi_hrtf_file_name, 'mbstoi_hrtf');
        %save(mat_file_name,'mbstoi_norm','mbstoi','equal_check');

        out_clean_wav_dir = strcat(out_dir,'\','clean.wav');
        %out_clean_hrtf_wav_dir = strcat(out_dir,'\','clean_hrtf.wav');
        out_mixed_wav_dir = strcat(out_dir,'\','mixed.wav');
        v_writewav(binaural_out,16000,out_mixed_wav_dir,[],[],[],[]);
        v_writewav(anechoic_speech,16000,out_clean_wav_dir,[],[],[],[]);
        %v_writewav(anechoic_speech_hrtf,16000,out_clean_hrtf_wav_dir,[],[],[],[]);
        fileID = fopen(strcat(out_dir,'\','log.txt'),'w');
        fprintf(fileID,'mbstoi.mat contains the full mbstoi array, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        fprintf(fileID,'The clean reference speech used was anechoic TIMIT speech \n');
        
        %fprintf(fileID,'mbstoi_hrtf.mat contains the full mbstoi array, using the clean_hrf speech as the clean reference speech, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        %fprintf(fileID,'The clean_hrtf reference speech is anechoic TIMIT speech convolved with the HRIR described below\n');
        %fprintf(fileID,'For both of these index 1 and 2 are using anechoic and reverberant speech as clean respectively \n');
        fprintf(fileID,'\n\n');
        fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{TIMIT_wav_index,1}),'\','/'),' \n'));
        fprintf(fileID,'SNR diffuse noise: %.1f',SNR_diff(SNR_diff_index));
        fprintf(fileID,'\n\n');

        fprintf(fileID,'Impulse Response Parameters: \n');
        fprintf(fileID,'Measured \n');
        fprintf(fileID,strcat(strrep(strcat('directory of measured impulse response used: ',measured_BRIR_wav_dirs{measured_BRIR_index,1}),'\','/'),' \n'));
        
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            fprintf(fileID,'Directional Noise Included \n');
            if direc_on(direc_index) == 1
                fprintf(fileID,'Number of directional sources: 1 \n');
            else
                fprintf(fileID,'Number of directional sources: 2 \n');
            end
            fprintf(fileID,'First directional source: \n');
            fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{1}));

            fprintf(fileID,'Measured \n');
            %fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index_direc(1)));
            %fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(1)));
            fprintf(fileID,strcat(strrep(strcat('directory of impulse response used: ',brir_out_dir_direc{1}),'\','/'),' \n'));
                

            if direc_type(direc_type_index{1}) == 0
                fprintf(fileID,'Non-speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{1}),'\','/'),' \n'));
            elseif direc_type(direc_type_index{1}) == 1
                fprintf(fileID,'Speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{1}{1}),'\','/'),' \n'));
            end
            if direc_on(direc_index) == 2
                fprintf(fileID,'Second directional source: \n');
                fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{2}));

                fprintf(fileID,'Measured \n');
                %fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index_direc{2}));
                %fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc{2}));
                fprintf(fileID,strcat(strrep(strcat('directory of impulse response used: ',brir_out_dir_direc{2}),'\','/'),' \n'));
                if direc_type(direc_type_index{2}) == 0
                    fprintf(fileID,'Non-speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{2}),'\','/'),' \n'));
                elseif direc_type(direc_type_index{2}) == 1
                    fprintf(fileID,'Speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{2}{1}),'\','/'),' \n'));
                end
            end
        end
        %}



        fclose(fileID);
        %measured_BRIR_pool = [];
        %{
        if ~isempty(TRAIN_strfind)
            TIMIT_train_file_count = TIMIT_train_file_count + 1;
        end
        if ~isempty(TEST_strfind)
            TIMIT_test_file_count = TIMIT_test_file_count + 1;
        end
        %}
        file_counter = file_counter + 1;


    end
end
    
    
