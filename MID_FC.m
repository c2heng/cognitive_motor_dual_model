

clear;clc
Prep = '/nii';
seses = {'ses-baseline','ses-followup2','ses-followup3'};
seses(randperm(length(seses))) = seses;
tasknames={'MID'};
tasknames(randperm(length(tasknames))) = tasknames;
%%
for sesid=1:3
    ses=seses{sesid};
    for tasknameid=1
        taskname=tasknames{tasknameid};
            
            output=['/FC/',taskname,'/',ses];
            mkdir(output);

            sub = dir(fullfile(Prep,'sub-*'));
            sub = sub([sub.isdir]);
            first_sub = zeros(1,length(sub));
            for i = 1:length(sub)

%                 disp(i)
                func = dir(fullfile(Prep,sub(i).name,ses,'func',['Preprocessed*task-' lower(taskname) '*.nii']));
                if ~isempty(func)
                    first_sub(i) = i;
                end
            end
            first_sub(first_sub == 0) = [];
            sub = sub(first_sub);
            disp(ses);
            disp(taskname);
            disp(size(sub));         
            t1_file = 'avg152T1.nii';
            tr = 2.2;
            

            roi_file = 'combined_masks.nii';
            roi_name = 'motor';

            
            %% batch preprocessing for single-subject single-session data
            
            parfor i=1:length(sub)
                try
                    
                    
                    spm_path = fullfile('/FC/SPM/',ses,sub(i).name,'SPM.mat');
                    
                    
                    out_path = fullfile(output,['conn_',sub(i).name]);
                    out_name = fullfile(output,['conn_',sub(i).name,'.mat']);
                    
                    if ~exist(out_path, 'dir')
                        mkdir(out_path);
                        fc_function(spm_path,t1_file,tr,out_name,roi_name,roi_file)
                        disp(['Process Subject_conn__',num2str(i,'%04d')]);
                        
                        movefile(fullfile(out_path,'/results/firstlevel/ANALYSIS_01/BETA*'),out_path);
                        movefile(fullfile(out_path,'/results/firstlevel/ANALYSIS_01/_list_conditions*'),out_path);
                        
                        rmdir(fullfile(out_path,'/data'),'s');
                        rmdir(fullfile(out_path,'/results/'),'s');
                        delete(out_name)    
                        nii_files = dir(fullfile(out_path, '*.nii'));
                        % Loop through each file and compress it
                        for file = 1:length(nii_files)
                            file_to_compress = fullfile(out_path, nii_files(file).name);
                            gzip(file_to_compress, out_path);
                            delete(file_to_compress);
                        end
                    else
                        
                        disp(['Donot Need to Process Subject_conn__',num2str(i,'%04d')]);
                        
                    end
                    
                catch ME
                    disp(['An error occurred: ', ME.message]);
                    
                end
            end
        
    end
end

function fc(spm_file,t1_file,tr,out_name,roi_name,roi_file)
    TR=tr;
    batch.filename=out_name;
    batch.Setup.spmfiles=spm_file;
    batch.Setup.structurals=cellstr(t1_file);
    batch.Setup.nsubjects=1;
    batch.Setup.RT=TR;
    batch.Setup.rois.names= cellstr(roi_name);
    batch.Setup.rois.files{1}=roi_file;
    batch.Setup.rois.mask = 0;           
    batch.Setup.rois.multiplelabels = 1; 
    batch.Setup.isnew=1;
    batch.Setup.done=1;
    batch.Setup.Setup.steps = [1 0 0 0];
    batch.Denoising.filter=[0.01,0.1];          
    batch.Denoising.done=1;
    batch.Denoising.despiking= 1;               
    batch.Analysis.modulation = 0;  
    batch.Analysis.measure=1;           
    batch.Analysis.weight=2;  
    batch.Analysis.type= 2;   
    batch.Analysis.sources={};
    batch.Analysis.done=1;
    
    conn_batch(batch);
    
    
